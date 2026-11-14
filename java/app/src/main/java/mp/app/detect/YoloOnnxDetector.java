package mp.app.detect;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** YOLO11 ONNX detector: letterbox 1280, output [1, 4+nc, 33600], class-aware NMS. Mirrors the Python reference. */
public final class YoloOnnxDetector implements AutoCloseable {
    public static final int S = 1280;

    public record Result(List<Detection> detections, double scale, int padLeft, int padTop, double inferenceMs) {}

    private record Coeffs(int[] min, double[][] k) {}

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final String inputName;
    private final List<String> names;
    private final float confThreshold;
    private final float iouThreshold;
    private final int maxDet;

    public YoloOnnxDetector(String modelPath, List<String> names, float conf, float iou, int maxDet) throws OrtException {
        this.names = List.copyOf(names);
        this.confThreshold = conf;
        this.iouThreshold = iou;
        this.maxDet = maxDet;
        OrtSession s;
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            so.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() - 2));
            s = env.createSession(modelPath, so);
        }
        this.session = s;
        this.inputName = s.getInputNames().iterator().next();
    }

    public Result detect(BufferedImage img) throws OrtException {
        int w = img.getWidth(), h = img.getHeight();
        double r = Math.min((double) S / w, (double) S / h);
        int nw = (int) Math.rint(w * r), nh = (int) Math.rint(h * r);
        int left = (int) Math.rint((S - nw) / 2.0 - 0.1), top = (int) Math.rint((S - nh) / 2.0 - 0.1);

        int[] argb = img.getRGB(0, 0, w, h, null, 0, w);
        float[] chw = new float[3 * S * S];
        Arrays.fill(chw, 114f / 255f);
        for (int c = 0; c < 3; c++) {
            int shift = 16 - 8 * c;
            int[] ch = new int[w * h];
            for (int i = 0; i < ch.length; i++) ch[i] = (argb[i] >> shift) & 0xFF;
            int[] rs = resize(ch, w, h, nw, nh);
            int base = c * S * S;
            for (int y = 0; y < nh; y++)
                for (int x = 0; x < nw; x++)
                    chw[base + (y + top) * S + (x + left)] = rs[y * nw + x] / 255f;
        }

        FloatBuffer fb = ByteBuffer.allocateDirect(chw.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        fb.put(chw);
        fb.rewind();
        long t0 = System.nanoTime();
        float[][] o;
        try (OnnxTensor t = OnnxTensor.createTensor(env, fb, new long[] {1, 3, S, S});
             OrtSession.Result res = session.run(Map.of(inputName, t))) {
            o = ((float[][][]) res.get(0).getValue())[0];
        }
        double ms = (System.nanoTime() - t0) / 1e6;

        int nc = o.length - 4, n = o[0].length;
        if (nc != names.size()) throw new IllegalStateException("Model has " + nc + " classes, expected " + names.size());
        int[] idx = new int[n], cl = new int[n];
        float[] cf = new float[n];
        int m = 0;
        for (int i = 0; i < n; i++) {
            int best = 0;
            float bs = o[4][i];
            for (int c = 1; c < nc; c++) { float v = o[4 + c][i]; if (v > bs) { bs = v; best = c; } }
            if (bs > confThreshold) { idx[m] = i; cf[m] = bs; cl[m] = best; m++; }
        }
        float[][] bx = new float[m][4];
        for (int k = 0; k < m; k++) {
            int i = idx[k];
            float cx = o[0][i], cy = o[1][i], bw = o[2][i], bh = o[3][i];
            bx[k][0] = cx - bw / 2; bx[k][1] = cy - bh / 2; bx[k][2] = cx + bw / 2; bx[k][3] = cy + bh / 2;
        }
        Integer[] order = new Integer[m];
        for (int k = 0; k < m; k++) order[k] = k;
        final float[] cfF = cf;
        Arrays.sort(order, (a, b) -> Float.compare(cfF[b], cfF[a]));

        List<Integer> keep = new ArrayList<>();
        for (int oi = 0; oi < m && keep.size() < maxDet; oi++) {
            int a = order[oi];
            boolean ok = true;
            for (int k : keep) {
                if (cl[k] == cl[a] && iou(bx[a], bx[k]) > iouThreshold) { ok = false; break; }
            }
            if (ok) keep.add(a);
        }

        List<Detection> out = new ArrayList<>(keep.size());
        float sc = (float) r;
        for (int a : keep) {
            float[] lb = bx[a];
            float[] im = {
                clamp((lb[0] - left) / sc, 0, w), clamp((lb[1] - top) / sc, 0, h),
                clamp((lb[2] - left) / sc, 0, w), clamp((lb[3] - top) / sc, 0, h)};
            out.add(new Detection(cl[a], names.get(cl[a]), cf[a], lb.clone(), im));
        }
        return new Result(out, r, left, top, ms);
    }

    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

    private static float iou(float[] a, float[] b) {
        float ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        float iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        float inter = ix * iy;
        float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        return ua <= 0 ? 0 : inter / ua;
    }

    /** Bilinear resize of one 8-bit channel, same filter and 8-bit rounding per pass as Pillow. */
    private static int[] resize(int[] src, int w, int h, int nw, int nh) {
        Coeffs cx = coeffs(w, nw), cy = coeffs(h, nh);
        int[] tmp = new int[nw * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < nw; x++) {
                int xmin = cx.min[x];
                double[] k = cx.k[x];
                double acc = 0;
                for (int j = 0; j < k.length; j++) acc += k[j] * src[y * w + xmin + j];
                tmp[y * nw + x] = clip(acc);
            }
        }
        int[] out = new int[nw * nh];
        for (int y = 0; y < nh; y++) {
            int ymin = cy.min[y];
            double[] k = cy.k[y];
            for (int x = 0; x < nw; x++) {
                double acc = 0;
                for (int j = 0; j < k.length; j++) acc += k[j] * tmp[(ymin + j) * nw + x];
                out[y * nw + x] = clip(acc);
            }
        }
        return out;
    }

    private static int clip(double v) { return Math.max(0, Math.min(255, (int) Math.floor(v + 0.5))); }

    private static Coeffs coeffs(int in, int out) {
        double scale = (double) in / out, fs = Math.max(scale, 1.0), support = fs;
        int[] min = new int[out];
        double[][] k = new double[out][];
        for (int i = 0; i < out; i++) {
            double center = (i + 0.5) * scale;
            int xmin = Math.max(0, (int) (center - support + 0.5));
            int xmax = Math.min(in, (int) (center + support + 0.5)) - xmin;
            double[] w = new double[Math.max(xmax, 0)];
            double sum = 0;
            for (int x = 0; x < w.length; x++) {
                double d = Math.abs((x + xmin - center + 0.5) / fs);
                w[x] = d < 1.0 ? 1.0 - d : 0.0;
                sum += w[x];
            }
            if (sum != 0) for (int x = 0; x < w.length; x++) w[x] /= sum;
            min[i] = xmin;
            k[i] = w;
        }
        return new Coeffs(min, k);
    }

    @Override
    public void close() throws OrtException { session.close(); }
}
