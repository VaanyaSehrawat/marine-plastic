package mp.fund;

import java.util.Random;

/** Hebbian learning: Oja's rule (Hebb with weight decay) converges to the first principal component. */
public final class Hebbian {
    private Hebbian() {}

    /** x: n samples x d features, already centered. Returns the learned weight vector. */
    public static double[] oja(Tensor x, double eta, int epochs, long seed) {
        int n = x.rows(), d = x.cols();
        Random rnd = new Random(seed);
        double[] w = new double[d];
        for (int i = 0; i < d; i++) w[i] = rnd.nextGaussian() * 0.1;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int e = 0; e < epochs; e++) {
            for (int i = n - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
            for (int idx : order) {
                double y = 0;
                for (int i = 0; i < d; i++) y += w[i] * x.get(idx, i);
                for (int i = 0; i < d; i++) w[i] += eta * y * (x.get(idx, i) - y * w[i]);
            }
        }
        return w;
    }
}
