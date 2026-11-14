package mp.fund;

import java.util.HashMap;
import java.util.Map;

/** First-order optimizers. Each parameter array has its own integer slot for state. */
public interface Optimizer {
    void update(int slot, double[] w, double[] g);

    final class Sgd implements Optimizer {
        private final double lr;

        public Sgd(double lr) { this.lr = lr; }

        @Override
        public void update(int slot, double[] w, double[] g) {
            for (int i = 0; i < w.length; i++) w[i] -= lr * g[i];
        }
    }

    final class Momentum implements Optimizer {
        private final double lr, beta;
        private final Map<Integer, double[]> vel = new HashMap<>();

        public Momentum(double lr, double beta) { this.lr = lr; this.beta = beta; }

        @Override
        public void update(int slot, double[] w, double[] g) {
            double[] v = vel.computeIfAbsent(slot, k -> new double[w.length]);
            for (int i = 0; i < w.length; i++) {
                v[i] = beta * v[i] + g[i];
                w[i] -= lr * v[i];
            }
        }
    }

    final class Adam implements Optimizer {
        private final double lr, b1, b2, eps;
        private final Map<Integer, double[]> m = new HashMap<>(), v = new HashMap<>();
        private final Map<Integer, int[]> t = new HashMap<>();

        public Adam(double lr) { this(lr, 0.9, 0.999, 1e-8); }

        public Adam(double lr, double b1, double b2, double eps) {
            this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps;
        }

        @Override
        public void update(int slot, double[] w, double[] g) {
            double[] mm = m.computeIfAbsent(slot, k -> new double[w.length]);
            double[] vv = v.computeIfAbsent(slot, k -> new double[w.length]);
            int[] tt = t.computeIfAbsent(slot, k -> new int[1]);
            tt[0]++;
            double c1 = 1.0 - Math.pow(b1, tt[0]), c2 = 1.0 - Math.pow(b2, tt[0]);
            for (int i = 0; i < w.length; i++) {
                mm[i] = b1 * mm[i] + (1.0 - b1) * g[i];
                vv[i] = b2 * vv[i] + (1.0 - b2) * g[i] * g[i];
                w[i] -= lr * (mm[i] / c1) / (Math.sqrt(vv[i] / c2) + eps);
            }
        }
    }
}
