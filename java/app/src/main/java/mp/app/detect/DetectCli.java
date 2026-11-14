package mp.app.detect;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.imageio.ImageIO;

/** Usage: DetectCli model.onnx image.png out.png [conf]. Draws boxes and prints per-class counts. */
public final class DetectCli {
    public static void main(String[] a) throws Exception {
        System.setProperty("java.awt.headless", "true");
        Path model = Path.of(a[0]), in = Path.of(a[1]), out = Path.of(a[2]);
        float conf = a.length > 3 ? Float.parseFloat(a[3]) : 0.25f;
        Map<String, Object> info = MiniJson.asObj(MiniJson.parse(Files.readString(model.resolveSibling("model_info.json"))));
        List<String> names = new ArrayList<>();
        for (Object o : MiniJson.asList(info.get("names"))) names.add((String) o);

        BufferedImage img = ImageIO.read(in.toFile());
        YoloOnnxDetector.Result r;
        try (YoloOnnxDetector det = new YoloOnnxDetector(model.toString(), names, conf, 0.7f, 400)) {
            det.detect(img);
            r = det.detect(img);
        }

        int w = img.getWidth(), h = img.getHeight();
        BufferedImage canvas = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.drawImage(img, 0, 0, null);
        g.setStroke(new BasicStroke(Math.max(1.5f, w / 500f)));
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, Math.max(11, w / 70)));
        Map<String, Integer> counts = new TreeMap<>();
        for (Detection d : r.detections()) {
            counts.merge(d.name(), 1, Integer::sum);
            Color c = Color.getHSBColor(d.cls() / (float) names.size(), 0.9f, 1f);
            g.setColor(c);
            float[] b = d.imageXyxy();
            g.drawRect(Math.round(b[0]), Math.round(b[1]), Math.round(b[2] - b[0]), Math.round(b[3] - b[1]));
            if (d.conf() >= 0.5f) g.drawString(String.format("%s %.2f", d.name(), d.conf()), Math.round(b[0]), Math.max(12, Math.round(b[1]) - 2));
        }
        g.dispose();
        ImageIO.write(canvas, "png", out.toFile());
        System.out.printf("%s: %d detections (conf>=%.2f) in %.0f ms | %s -> %s%n", in.getFileName(), r.detections().size(), conf, r.inferenceMs(), counts, out);
    }
}
