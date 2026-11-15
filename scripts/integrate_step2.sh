#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
rm -f build/step2.ok
[ -f build/step1.ok ] || { echo "STEP 2 BLOCKED: Step 1 has not reported READY (build/step1.ok missing)"; exit 1; }
R=java/app/src/main/java/mp/app/route; mkdir -p "$R"
put() { if [ -s "$1" ]; then echo "  keeping existing $1"; cat > /dev/null; else cat > "$1"; echo "  created $1"; fi; }

put "$R/RouteScorer.java" <<'JEOF'
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
JEOF

put "$R/RouteTest.java" <<'JEOF'
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
JEOF

put scripts/test_route.sh <<'JEOF'
#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 1
JH="$ROOT/.tools/jdk-21/Contents/Home"; JAVAC="$JH/bin/javac"; JAVA="$JH/bin/java"
JAR="$ROOT/java/lib/onnxruntime-1.20.0.jar"
M="$ROOT/models/yolo11m_1280/yolo11m_1280_best.onnx"; MF="$ROOT/models/yolo11m_1280/fixture"
RM="$ROOT/models/route/route_model.onnx"; RF="$ROOT/models/route/route_fixture.json"
for f in "$JAVAC" "$JAR" "$M" "$MF/fixture.json" "$ROOT/models/yolo11m_1280/model_info.json" "$RM" "$RF"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
OUT="$ROOT/java/out"; mkdir -p "$OUT"; J=java/app/src/main/java/mp/app
"$JAVAC" -encoding UTF-8 -cp "$JAR" -d "$OUT" $J/detect/*.java $J/route/*.java $J/planner/*.java $J/server/*.java || { echo "COMPILE FAILED"; exit 1; }
echo "COMPILED OK"
CP="$OUT:$JAR"
echo "== YOLO11m fixture test =="
"$JAVA" -Xmx3g -Djava.awt.headless=true -cp "$CP" mp.app.detect.FixtureTest "$M" "$MF" || exit 1
echo "== Route transformer test =="
"$JAVA" -Xmx3g -Djava.awt.headless=true -cp "$CP" mp.app.route.RouteTest "$RM" "$RF" "$M" "$MF" || exit 1
JEOF

for f in "$R/RouteScorer.java" "$R/RouteTest.java" scripts/test_route.sh; do
  [ -s "$f" ] || { echo "STEP 2 FAILED: $f missing or empty"; exit 1; }
done
chmod +x scripts/test_route.sh
bash scripts/test_route.sh; rc=$?
if [ $rc -eq 0 ]; then touch build/step2.ok; echo; echo "STEP 2: PASS"; else echo; echo "STEP 2: FAIL (exit code $rc)"; fi
