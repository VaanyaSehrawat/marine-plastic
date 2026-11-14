package mp.fund;

import java.util.Random;

/** Fully connected network with softmax cross-entropy output and manual backpropagation. */
public final class Mlp {
    public enum Activation { RELU, TANH, SIGMOID }

    private final Activation hidden;
    private final Tensor[] w, b;

    public Mlp(int[] sizes, Activation hidden, long seed) {
        if (sizes.length < 2) throw new IllegalArgumentException("Need input and output sizes");
        this.hidden = hidden;
        int layers = sizes.length - 1;
        w = new Tensor[layers];
        b = new Tensor[layers];
        Random rnd = new Random(seed);
        for (int l = 0; l < layers; l++) {
            double scale = hidden == Activation.RELU ? Math.sqrt(2.0 / sizes[l]) : Math.sqrt(2.0 / (sizes[l] + sizes[l + 1]));
            w[l] = Tensor.randn(rnd, scale, sizes[l], sizes[l + 1]);
            b[l] = new Tensor(1, sizes[l + 1]);
        }
    }

    public int layers() { return w.length; }
    public Tensor weight(int l) { return w[l]; }
    public Tensor bias(int l) { return b[l]; }

    private Tensor activate(Tensor z) {
        return switch (hidden) {
            case RELU -> z.map(v -> v > 0 ? v : 0.0);
            case TANH -> z.map(Math::tanh);
            case SIGMOID -> z.map(v -> 1.0 / (1.0 + Math.exp(-v)));
        };
    }

    /** Derivative expressed through the activation output a. */
    private Tensor derivative(Tensor a) {
        return switch (hidden) {
            case RELU -> a.map(v -> v > 0 ? 1.0 : 0.0);
            case TANH -> a.map(v -> 1.0 - v * v);
            case SIGMOID -> a.map(v -> v * (1.0 - v));
        };
    }

    private Tensor[] forward(Tensor x) {
        Tensor[] a = new Tensor[w.length + 1];
        a[0] = x;
        for (int l = 0; l < w.length; l++) {
            Tensor z = a[l].matmul(w[l]).addRow(b[l]);
            a[l + 1] = (l == w.length - 1) ? z : activate(z);
        }
        return a;
    }

    public Tensor logits(Tensor x) {
        Tensor[] a = forward(x);
        return a[a.length - 1];
    }

    private static double softmaxXent(Tensor logits, int[] y, Tensor delta) {
        int n = logits.rows(), k = logits.cols();
        double loss = 0;
        double[] p = new double[k];
        for (int i = 0; i < n; i++) {
            double mx = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) mx = Math.max(mx, logits.get(i, j));
            double sum = 0;
            for (int j = 0; j < k; j++) { p[j] = Math.exp(logits.get(i, j) - mx); sum += p[j]; }
            for (int j = 0; j < k; j++) p[j] /= sum;
            loss -= Math.log(Math.max(p[y[i]], 1e-300));
            if (delta != null)
                for (int j = 0; j < k; j++) delta.set(i, j, (p[j] - (j == y[i] ? 1.0 : 0.0)) / n);
        }
        return loss / n;
    }

    public double loss(Tensor x, int[] y) { return softmaxXent(logits(x), y, null); }

    /** Fills gw and gb (length layers()) and returns the mean cross-entropy loss. */
    public double computeGradients(Tensor x, int[] y, Tensor[] gw, Tensor[] gb) {
        Tensor[] a = forward(x);
        int last = w.length;
        Tensor logits = a[last];
        Tensor delta = new Tensor(logits.rows(), logits.cols());
        double loss = softmaxXent(logits, y, delta);
        for (int l = last - 1; l >= 0; l--) {
            gw[l] = a[l].transpose().matmul(delta);
            gb[l] = delta.sumRows();
            if (l > 0) {
                Tensor da = delta.matmul(w[l].transpose());
                delta = da.mul(derivative(a[l]));
            }
        }
        return loss;
    }

    public double trainStep(Tensor x, int[] y, Optimizer opt) {
        Tensor[] gw = new Tensor[w.length], gb = new Tensor[w.length];
        double loss = computeGradients(x, y, gw, gb);
        for (int l = 0; l < w.length; l++) {
            opt.update(2 * l, w[l].data(), gw[l].data());
            opt.update(2 * l + 1, b[l].data(), gb[l].data());
        }
        return loss;
    }

    public int[] predict(Tensor x) {
        Tensor lg = logits(x);
        int[] r = new int[lg.rows()];
        for (int i = 0; i < r.length; i++) r[i] = lg.argmaxRow(i);
        return r;
    }

    public double accuracy(Tensor x, int[] y) {
        int[] p = predict(x);
        int ok = 0;
        for (int i = 0; i < p.length; i++) if (p[i] == y[i]) ok++;
        return (double) ok / p.length;
    }
}
