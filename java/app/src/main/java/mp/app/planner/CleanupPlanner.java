package mp.app.planner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import mp.app.detect.Detection;

/** Image-space cleanup planner: tile scores from detections, inspection route over high-scoring tiles. Not GPS navigation. */
public final class CleanupPlanner {
    /** Heuristic priority weights (configurable assumption, not a measured quantity). */
    public static final Map<String, Double> WEIGHT = Map.ofEntries(
        Map.entry("fishing_net", 3.0), Map.entry("other_fishing_gear", 3.0), Map.entry("buoy", 2.5),
        Map.entry("pet_bottle", 2.0), Map.entry("other_bottle", 2.0), Map.entry("plastic_bag", 2.0),
        Map.entry("box_shaped_case", 2.0), Map.entry("other_container", 2.0), Map.entry("rope", 2.0),
        Map.entry("other_string", 1.5), Map.entry("styrene_foam", 1.5), Map.entry("others", 1.0),
        Map.entry("fragment", 0.5));

    public record Plan(int rows, int cols, double[][] score, int[][] count, List<int[]> route,
                       double plannedLen, double serpentineLen, int selected) {}

    private CleanupPlanner() {}

    public static Plan plan(List<Detection> dets, int w, int h, int rows, int cols, double threshold) {
        double[][] sc = new double[rows][cols];
        int[][] cnt = new int[rows][cols];
        for (Detection d : dets) {
            float[] b = d.imageXyxy();
            double cx = (b[0] + b[2]) / 2.0, cy = (b[1] + b[3]) / 2.0;
            int c = (int) Math.max(0, Math.min(cols - 1, Math.floor(cx / w * cols)));
            int r = (int) Math.max(0, Math.min(rows - 1, Math.floor(cy / h * rows)));
            sc[r][c] += WEIGHT.getOrDefault(d.name(), 1.0) * d.conf();
            cnt[r][c]++;
        }
        List<int[]> sel = new ArrayList<>();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (sc[r][c] >= threshold && cnt[r][c] > 0) sel.add(new int[] {r, c});
        List<int[]> route = twoOpt(nearestNeighbour(sel));
        return new Plan(rows, cols, sc, cnt, route, length(route), length(serpentine(sel)), sel.size());
    }

    /** Externally supplied tile scores (for example from the learned route model). cnt is only used for display. */
    public static Plan planFromScores(double[][] sc, int[][] cnt, int rows, int cols, double threshold) {
        List<int[]> sel = new ArrayList<>();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (sc[r][c] >= threshold) sel.add(new int[] {r, c});
        List<int[]> route = twoOpt(nearestNeighbour(sel));
        return new Plan(rows, cols, sc, cnt, route, length(route), length(serpentine(sel)), sel.size());
    }

    private static double d(int[] a, int[] b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

    /** Path length in tile units, starting from the entry tile (0,0). */
    public static double length(List<int[]> route) {
        double s = 0;
        int[] prev = {0, 0};
        for (int[] t : route) { s += d(prev, t); prev = t; }
        return s;
    }

    public static List<int[]> nearestNeighbour(List<int[]> sel) {
        List<int[]> rem = new ArrayList<>(sel), out = new ArrayList<>();
        int[] cur = {0, 0};
        while (!rem.isEmpty()) {
            int best = 0;
            double bd = Double.MAX_VALUE;
            for (int i = 0; i < rem.size(); i++) {
                double dd = d(cur, rem.get(i));
                if (dd < bd - 1e-12) { bd = dd; best = i; }
            }
            cur = rem.remove(best);
            out.add(cur);
        }
        return out;
    }

    public static List<int[]> twoOpt(List<int[]> route) {
        List<int[]> L = new ArrayList<>();
        L.add(new int[] {0, 0});
        L.addAll(route);
        int n = L.size();
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int i = 1; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    double before = d(L.get(i - 1), L.get(i)) + (j + 1 < n ? d(L.get(j), L.get(j + 1)) : 0);
                    double after = d(L.get(i - 1), L.get(j)) + (j + 1 < n ? d(L.get(i), L.get(j + 1)) : 0);
                    if (after < before - 1e-9) { Collections.reverse(L.subList(i, j + 1)); improved = true; }
                }
            }
        }
        return new ArrayList<>(L.subList(1, n));
    }

    /** Baseline: row-by-row serpentine sweep restricted to the same selected tiles. */
    public static List<int[]> serpentine(List<int[]> sel) {
        List<int[]> out = new ArrayList<>(sel);
        out.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : (a[0] % 2 == 0 ? Integer.compare(a[1], b[1]) : Integer.compare(b[1], a[1])));
        return out;
    }
}
