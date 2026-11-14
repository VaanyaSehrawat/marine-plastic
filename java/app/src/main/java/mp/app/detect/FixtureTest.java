package mp.app.detect;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;

/** Checks the Java detector against fixture.json produced by the PyTorch/ONNX Python reference. */
public final class FixtureTest {
    public static void main(String[] a) throws Exception {
        System.setProperty("java.awt.headless", "true");
        Path model = Path.of(a[0]), dir = Path.of(a[1]);
        Map<String, Object> fx = MiniJson.asObj(MiniJson.parse(Files.readString(dir.resolve("fixture.json"))));
        List<String> names = new ArrayList<>();
        for (Object o : MiniJson.asList(fx.get("names"))) names.add((String) o);
        float conf = (float) MiniJson.num(fx.get("conf")), iou = (float) MiniJson.num(fx.get("iou"));
        int maxDet = (int) MiniJson.num(fx.get("max_det"));
        boolean allOk = true;

        try (YoloOnnxDetector det = new YoloOnnxDetector(model.toString(), names, conf, iou, maxDet)) {
            System.out.println("Model loaded: classes=" + names.size() + " conf=" + conf + " iou=" + iou + " maxDet=" + maxDet);
            for (Object io : MiniJson.asList(fx.get("images"))) {
                Map<String, Object> im = MiniJson.asObj(io);
                String file = (String) im.get("file");
                BufferedImage img = ImageIO.read(dir.resolve(file).toFile());
                det.detect(img); // warm-up
                YoloOnnxDetector.Result r = det.detect(img);

                double expScale = MiniJson.num(im.get("scale"));
                int expLeft = (int) MiniJson.num(im.get("pad_left")), expTop = (int) MiniJson.num(im.get("pad_top"));
                int expN = (int) MiniJson.num(im.get("n_detections"));
                boolean geomOk = Math.abs(r.scale() - expScale) < 1e-6 && r.padLeft() == expLeft && r.padTop() == expTop;
                boolean countOk = Math.abs(r.detections().size() - expN) <= 2;

                int matched = 0;
                List<Object> exp = MiniJson.asList(im.get("top10"));
                for (Object eo : exp) {
                    Map<String, Object> e = MiniJson.asObj(eo);
                    int ec = (int) MiniJson.num(e.get("cls"));
                    float econf = (float) MiniJson.num(e.get("conf"));
                    List<Object> eb = MiniJson.asList(e.get("box_letterbox_xyxy"));
                    boolean hit = false;
                    for (Detection d : r.detections()) {
                        if (d.cls() != ec || Math.abs(d.conf() - econf) > 0.02f) continue;
                        double md = 0;
                        for (int i = 0; i < 4; i++) md = Math.max(md, Math.abs(d.letterboxXyxy()[i] - MiniJson.num(eb.get(i))));
                        if (md <= 3.0) { hit = true; break; }
                    }
                    if (hit) matched++;
                }
                boolean topOk = matched == exp.size();
                allOk &= geomOk && countOk && topOk;
                System.out.printf("%s: %dx%d scale=%.4f pad=(%d,%d) geometry=%s | dets java=%d expected=%d (%s) | top10 matched %d/%d | infer %.0f ms%n",
                    file, img.getWidth(), img.getHeight(), r.scale(), r.padLeft(), r.padTop(), geomOk ? "OK" : "BAD",
                    r.detections().size(), expN, countOk ? "OK" : "BAD", matched, exp.size(), r.inferenceMs());
            }
        }
        System.out.println(allOk ? "FIXTURE TEST: PASS" : "FIXTURE TEST: FAIL");
        System.exit(allOk ? 0 : 1);
    }
}
