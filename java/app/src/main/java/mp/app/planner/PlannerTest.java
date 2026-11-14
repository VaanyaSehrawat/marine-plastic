package mp.app.planner;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import mp.app.detect.Detection;

public final class PlannerTest {
    private static int checks = 0;

    private static void check(boolean ok, String msg) {
        checks++;
        if (!ok) throw new AssertionError("PLANNER TEST FAILED: " + msg);
    }

    private static Set<String> keys(List<int[]> l) {
        Set<String> s = new HashSet<>();
        for (int[] t : l) s.add(t[0] + "," + t[1]);
        return s;
    }

    public static void main(String[] a) {
        // 1. known optimal route on a line
        List<int[]> sel = List.of(new int[] {0, 3}, new int[] {0, 1}, new int[] {0, 2});
        List<int[]> r = CleanupPlanner.twoOpt(CleanupPlanner.nearestNeighbour(sel));
        check(r.size() == 3 && r.get(0)[1] == 1 && r.get(1)[1] == 2 && r.get(2)[1] == 3, "line order");
        check(Math.abs(CleanupPlanner.length(r) - 3.0) < 1e-9, "line length");

        // 2. detection to tile assignment and class weight
        Detection d = new Detection(7, "fishing_net", 0.8f, new float[4], new float[] {300, 200, 400, 300});
        CleanupPlanner.Plan p = CleanupPlanner.plan(List.of(d), 800, 600, 6, 8, 1.0);
        check(p.count()[2][3] == 1 && Math.abs(p.score()[2][3] - 2.4) < 1e-6, "tile assignment and weight");
        check(p.selected() == 1 && p.route().size() == 1, "single selected tile");

        // 3. threshold excludes weak tiles
        Detection weak = new Detection(12, "fragment", 0.3f, new float[4], new float[] {10, 10, 20, 20});
        check(CleanupPlanner.plan(List.of(weak), 800, 600, 6, 8, 1.0).selected() == 0, "threshold excludes weak tile");

        // 4. random trials: valid permutation and 2-opt never longer than nearest neighbour
        Random rnd = new Random(42);
        for (int t = 0; t < 300; t++) {
            List<int[]> s = new ArrayList<>();
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 8; j++)
                    if (rnd.nextDouble() < 0.35) s.add(new int[] {i, j});
            List<int[]> nn = CleanupPlanner.nearestNeighbour(s), opt = CleanupPlanner.twoOpt(nn);
            check(opt.size() == s.size() && keys(opt).equals(keys(s)), "route is a permutation of selected tiles");
            check(CleanupPlanner.length(opt) <= CleanupPlanner.length(nn) + 1e-9, "2-opt not longer than nearest neighbour");
        }
        System.out.println("PLANNER TEST: PASS (" + checks + " checks)");
    }
}
