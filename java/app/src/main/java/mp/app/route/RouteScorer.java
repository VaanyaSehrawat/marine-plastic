package mp.app.route;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.util.List;
import java.util.Map;
import mp.app.detect.Detection;

/** Route-context transformer: per-tile detection features (6x8 grid) to per-tile priority, run with ONNX Runtime. */
public final class RouteScorer implements AutoCloseable {
    public static final int R = 6, C = 8, T = R * C, NC = 13, NF = NC + 2;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final String inputName;

    public RouteScorer(String modelPath) throws OrtException {
        OrtSession s;
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            s = env.createSession(modelPath, so);
        }
        this.session = s;
        this.inputName = s.getInputNames().iterator().next();
    }

    /** Same features as the Python training code: log1p of per-class confidence sums, detection count, max confidence. */
    public static float[][] features(List<Detection> dets, int w, int h, float cmin) {
        float[][] f = new float[T][NF];
        for (Detection d : dets) {
            if (d.conf() < cmin) continue;
            float[] b = d.imageXyxy();
            double cx = (b[0] + b[2]) / 2.0, cy = (b[1] + b[3]) / 2.0;
            int c = (int) Math.min(C - 1, Math.max(0, Math.floor(cx / w * C)));
            int r = (int) Math.min(R - 1, Math.max(0, Math.floor(cy / h * R)));
            int t = r * C + c;
            f[t][d.cls()] += d.conf();
            f[t][NC] += 1f;
            f[t][NC + 1] = Math.max(f[t][NC + 1], d.conf());
        }
        for (int t = 0; t < T; t++)
            for (int k = 0; k < NF; k++) f[t][k] = (float) Math.log1p(f[t][k]);
        return f;
    }

    /** Returns 48 priorities on the log1p scale (row-major tiles). */
    public float[] priority(float[][] tiles) throws OrtException {
        float[][][] in = new float[1][][];
        in[0] = tiles;
        try (OnnxTensor t = OnnxTensor.createTensor(env, in);
             OrtSession.Result res = session.run(Map.of(inputName, t))) {
            return ((float[][]) res.get(0).getValue())[0];
        }
    }

    @Override
    public void close() throws OrtException { session.close(); }
}
