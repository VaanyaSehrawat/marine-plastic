package mp.app.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import javax.imageio.ImageIO;
import mp.app.detect.Detection;
import mp.app.detect.MiniJson;
import mp.app.detect.YoloOnnxDetector;
import mp.app.planner.CleanupPlanner;

/** Local Java-served dashboard: GET /, GET /api/samples, GET /sample/{file}, POST /api/analyze. */
public final class DashboardServer {
    private static YoloOnnxDetector detector;
    private static Path fixtureDir, htmlPath;

    public static void main(String[] a) throws Exception {
        System.setProperty("java.awt.headless", "true");
        Path model = Path.of(a[0]);
        fixtureDir = Path.of(a[1]);
        htmlPath = Path.of(a[2]);
        int port = Integer.parseInt(a[3]);
        Map<String, Object> info = MiniJson.asObj(MiniJson.parse(Files.readString(model.resolveSibling("model_info.json"))));
        List<String> names = new ArrayList<>();
        for (Object o : MiniJson.asList(info.get("names"))) names.add((String) o);
        detector = new YoloOnnxDetector(model.toString(), names, 0.10f, 0.7f, 400);
        HttpServer srv = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        srv.createContext("/", DashboardServer::route);
        srv.start();
        System.out.println("Dashboard running at http://127.0.0.1:" + port + "  (Ctrl+C to stop)");
    }

    private static void route(HttpExchange ex) throws IOException {
        try {
            String path = ex.getRequestURI().getPath(), method = ex.getRequestMethod();
            if (method.equals("GET") && (path.equals("/") || path.equals("/index.html"))) {
                send(ex, 200, "text/html; charset=utf-8", Files.readAllBytes(htmlPath));
            } else if (method.equals("GET") && path.equals("/api/samples")) {
                List<String> l = new ArrayList<>();
                try (Stream<Path> s = Files.list(fixtureDir)) {
                    s.map(p -> p.getFileName().toString()).filter(n -> n.matches("[A-Za-z0-9_.-]+\\.(png|jpg|jpeg)")).forEach(l::add);
                }
                Collections.sort(l);
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < l.size(); i++) sb.append(i > 0 ? "," : "").append('"').append(esc(l.get(i))).append('"');
                send(ex, 200, "application/json", sb.append("]").toString().getBytes(StandardCharsets.UTF_8));
            } else if (method.equals("GET") && path.startsWith("/sample/")) {
                Path p = samplePath(path.substring(8));
                send(ex, 200, p.toString().endsWith(".png") ? "image/png" : "image/jpeg", Files.readAllBytes(p));
            } else if (method.equals("POST") && path.equals("/api/analyze")) {
                analyze(ex);
            } else {
                send(ex, 404, "text/plain", "Not found".getBytes(StandardCharsets.UTF_8));
            }
        } catch (IllegalArgumentException e) {
            send(ex, 400, "application/json", ("{\"error\":\"" + esc(String.valueOf(e.getMessage())) + "\"}").getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            e.printStackTrace();
            send(ex, 500, "application/json", ("{\"error\":\"" + esc(String.valueOf(e.getMessage())) + "\"}").getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void analyze(HttpExchange ex) throws Exception {
        Map<String, String> q = query(ex);
        float conf = (float) clamp(num(q.get("conf"), 0.25), 0.10, 0.90);
        int rows = (int) clamp(num(q.get("rows"), 6), 2, 20), cols = (int) clamp(num(q.get("cols"), 8), 2, 20);
        double thr = clamp(num(q.get("thr"), 1.0), 0, 50);
        byte[] body = ex.getRequestBody().readAllBytes();
        BufferedImage img = q.containsKey("sample")
            ? ImageIO.read(samplePath(q.get("sample")).toFile())
            : ImageIO.read(new ByteArrayInputStream(body));
        if (img == null) throw new IllegalArgumentException("Could not decode image (use PNG or JPEG)");
        int w = img.getWidth(), h = img.getHeight();
        YoloOnnxDetector.Result r;
        synchronized (detector) { r = detector.detect(img); }
        List<Detection> dets = new ArrayList<>();
        for (Detection d : r.detections()) if (d.conf() >= conf) dets.add(d);
        CleanupPlanner.Plan plan = CleanupPlanner.plan(dets, w, h, rows, cols, thr);

        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\"width\":").append(w).append(",\"height\":").append(h)
          .append(",\"inferMs\":").append(f(r.inferenceMs())).append(",\"conf\":").append(f(conf))
          .append(",\"nDet\":").append(dets.size()).append(",\"detections\":[");
        Map<String, Integer> cc = new TreeMap<>();
        for (int i = 0; i < dets.size(); i++) {
            Detection d = dets.get(i);
            float[] b = d.imageXyxy();
            cc.merge(d.name(), 1, Integer::sum);
            sb.append(i > 0 ? "," : "").append("{\"c\":").append(d.cls()).append(",\"n\":\"").append(esc(d.name()))
              .append("\",\"p\":").append(String.format(Locale.ROOT, "%.4f", d.conf()))
              .append(",\"b\":[").append(String.format(Locale.ROOT, "%.1f,%.1f,%.1f,%.1f", b[0], b[1], b[2], b[3])).append("]}");
        }
        sb.append("],\"classCounts\":{");
        int k = 0;
        for (Map.Entry<String, Integer> e : cc.entrySet())
            sb.append(k++ > 0 ? "," : "").append('"').append(esc(e.getKey())).append("\":").append(e.getValue());
        sb.append("},\"grid\":{\"rows\":").append(rows).append(",\"cols\":").append(cols).append(",\"score\":[");
        for (int i = 0; i < rows; i++) {
            sb.append(i > 0 ? "," : "").append("[");
            for (int j = 0; j < cols; j++) sb.append(j > 0 ? "," : "").append(f(plan.score()[i][j]));
            sb.append("]");
        }
        sb.append("],\"count\":[");
        for (int i = 0; i < rows; i++) {
            sb.append(i > 0 ? "," : "").append("[");
            for (int j = 0; j < cols; j++) sb.append(j > 0 ? "," : "").append(plan.count()[i][j]);
            sb.append("]");
        }
        sb.append("]},\"route\":[");
        for (int i = 0; i < plan.route().size(); i++)
            sb.append(i > 0 ? "," : "").append("[").append(plan.route().get(i)[0]).append(",").append(plan.route().get(i)[1]).append("]");
        double saved = plan.serpentineLen() > 1e-9 ? (plan.serpentineLen() - plan.plannedLen()) / plan.serpentineLen() * 100.0 : 0.0;
        sb.append("],\"metrics\":{\"selected\":").append(plan.selected())
          .append(",\"planned\":").append(f(plan.plannedLen())).append(",\"serpentine\":").append(f(plan.serpentineLen()))
          .append(",\"savedPct\":").append(f(saved)).append(",\"threshold\":").append(f(thr)).append("}}");
        send(ex, 200, "application/json", sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static Path samplePath(String name) throws IOException {
        if (name == null || !name.matches("[A-Za-z0-9_.-]+\\.(png|jpg|jpeg)")) throw new IllegalArgumentException("Bad sample name");
        Path p = fixtureDir.resolve(name);
        if (!Files.isRegularFile(p)) throw new IllegalArgumentException("Sample not found: " + name);
        return p;
    }

    private static Map<String, String> query(HttpExchange ex) {
        Map<String, String> m = new HashMap<>();
        String q = ex.getRequestURI().getRawQuery();
        if (q == null) return m;
        for (String kv : q.split("&")) {
            int i = kv.indexOf('=');
            if (i > 0) m.put(URLDecoder.decode(kv.substring(0, i), StandardCharsets.UTF_8), URLDecoder.decode(kv.substring(i + 1), StandardCharsets.UTF_8));
        }
        return m;
    }

    private static void send(HttpExchange ex, int code, String type, byte[] bytes) throws IOException {
        ex.getResponseHeaders().set("Content-Type", type);
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private static String f(double v) { return String.format(Locale.ROOT, "%.3f", v); }
    private static String esc(String s) { return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " "); }
    private static double clamp(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }
    private static double num(String s, double def) {
        try { return s == null ? def : Double.parseDouble(s); } catch (NumberFormatException e) { return def; }
    }
}
