package mp.app.route;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import mp.app.detect.Detection;
import mp.app.detect.MiniJson;
import mp.app.detect.YoloOnnxDetector;

/** Args: route_model.onnx route_fixture.json yolo.onnx yolo_fixture_dir. */
public final class RouteTest {
    private static int[] top5(float[] p) {
        Integer[] o = new Integer[p.length];
        for (int i = 0; i < o.length; i++) o[i] = i;
        Arrays.sort(o, (x, y) -> Float.compare(p[y], p[x]));
        int[] r = new int[5];
        for (int i = 0; i < 5; i++) r[i] = o[i];
        return r;
    }

    private static double pearson(float[] a, float[] b) {
        double ma = 0, mb = 0;
        for (int i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.length; mb /= a.length;
        double sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return sab / Math.sqrt(saa * sbb + 1e-30);
    }

    public static void main(String[] a) throws Exception {
        System.setProperty("java.awt.headless", "true");
        Path ym = Path.of(a[2]), yd = Path.of(a[3]);
        Map<String, Object> info = MiniJson.asObj(MiniJson.parse(Files.readString(ym.resolveSibling("model_info.json"))));
        List<String> names = new ArrayList<>();
        for (Object o : MiniJson.asList(info.get("names"))) names.add((String) o);
        List<Object> fx = MiniJson.asList(MiniJson.parse(Files.readString(Path.of(a[1]))));
        boolean ok = true;
        try (RouteScorer rs = new RouteScorer(a[0]);
             YoloOnnxDetector det = new YoloOnnxDetector(ym.toString(), names, 0.10f, 0.7f, 400)) {
            for (Object fo : fx) {
                Map<String, Object> e = MiniJson.asObj(fo);
                String file = (String) e.get("file");
                int w = (int) MiniJson.num(e.get("w")), h = (int) MiniJson.num(e.get("h"));
                List<Detection> dets = new ArrayList<>();
                for (Object d : MiniJson.asList(e.get("dets"))) {
                    List<Object> q = MiniJson.asList(d);
                    int k = (int) MiniJson.num(q.get(5));
                    dets.add(new Detection(k, names.get(k), (float) MiniJson.num(q.get(4)), new float[4],
                        new float[] {(float) MiniJson.num(q.get(0)), (float) MiniJson.num(q.get(1)),
                                     (float) MiniJson.num(q.get(2)), (float) MiniJson.num(q.get(3))}));
                }
                float[][] tiles = RouteScorer.features(dets, w, h, 0.10f);
                List<Object> expT = MiniJson.asList(e.get("tiles"));
                double fd = 0;
                for (int t = 0; t < RouteScorer.T; t++) {
                    List<Object> row = MiniJson.asList(expT.get(t));
                    for (int k = 0; k < RouteScorer.NF; k++) fd = Math.max(fd, Math.abs(tiles[t][k] - MiniJson.num(row.get(k))));
                }
                List<Object> expP = MiniJson.asList(e.get("priority"));
                float[] exp = new float[RouteScorer.T];
                for (int t = 0; t < exp.length; t++) exp[t] = (float) MiniJson.num(expP.get(t));
                float[] got = rs.priority(tiles);
                double pd = 0;
                for (int t = 0; t < exp.length; t++) pd = Math.max(pd, Math.abs(got[t] - exp[t]));
                boolean partA = fd < 1e-4 && pd < 1e-4;

                BufferedImage img = ImageIO.read(yd.resolve(file).toFile());
                YoloOnnxDetector.Result r = det.detect(img);
                float[] e2e = rs.priority(RouteScorer.features(r.detections(), img.getWidth(), img.getHeight(), 0.10f));
                double cor = pearson(e2e, exp);
                int[] ta = top5(e2e), tb = top5(exp);
                int ov = 0;
                for (int x : ta) for (int y : tb) if (x == y) ov++;
                boolean partB = cor >= 0.95;
                ok &= partA && partB;
                System.out.printf("%s: features max diff %.1e | transformer max diff %.1e (%s) | end-to-end: dets java=%d python=%d, priority correlation %.4f, top-5 overlap %d/5 (%s)%n",
                    file, fd, pd, partA ? "OK" : "BAD", r.detections().size(), MiniJson.asList(e.get("dets")).size(), cor, ov, partB ? "OK" : "BAD");
            }
        }
        System.out.println(ok ? "ROUTE TEST: PASS" : "ROUTE TEST: FAIL");
        System.exit(ok ? 0 : 1);
    }
}
