package mp.fund;

import java.util.Random;

/** Winner-take-all competitive learning with farthest-point initialisation and a decaying learning rate. */
public final class Competitive {
    private final double[][] protos;

    private Competitive(double[][] protos) { this.protos = protos; }

    public double[][] prototypes() { return protos; }

    public int winner(double[] x) {
        int best = 0;
        double bd = Double.MAX_VALUE;
        for (int k = 0; k < protos.length; k++) {
            double d = 0;
            for (int j = 0; j < x.length; j++) { double t = x[j] - protos[k][j]; d += t * t; }
            if (d < bd) { bd = d; best = k; }
        }
        return best;
    }

    private static double[] row(Tensor x, int i) {
        double[] r = new double[x.cols()];
        for (int j = 0; j < r.length; j++) r[j] = x.get(i, j);
        return r;
    }

    private static double dist2(double[] a, double[] b) {
        double d = 0;
        for (int j = 0; j < a.length; j++) { double t = a[j] - b[j]; d += t * t; }
        return d;
    }

    public static Competitive train(Tensor x, int k, double lr0, int epochs, long seed) {
        int n = x.rows();
        Random rnd = new Random(seed);
        double[][] p = new double[k][];
        p[0] = row(x, rnd.nextInt(n));
        for (int c = 1; c < k; c++) {
            int best = 0;
            double bd = -1;
            for (int i = 0; i < n; i++) {
                double[] xi = row(x, i);
                double md = Double.MAX_VALUE;
                for (int q = 0; q < c; q++) md = Math.min(md, dist2(p[q], xi));
                if (md > bd) { bd = md; best = i; }
            }
            p[c] = row(x, best);
        }
        Competitive model = new Competitive(p);
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int e = 0; e < epochs; e++) {
            double lr = Math.max(0.005, lr0 * (1.0 - (double) e / epochs));
            for (int i = n - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
            for (int idx : order) {
                double[] xi = row(x, idx);
                int w = model.winner(xi);
                for (int j = 0; j < xi.length; j++) p[w][j] += lr * (xi[j] - p[w][j]);
            }
        }
        return model;
    }
}
