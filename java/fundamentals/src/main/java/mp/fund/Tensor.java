package mp.fund;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/** Dense row-major tensor of doubles. Storage is rank-N; linear algebra methods require rank 2. */
public final class Tensor {
    private final int[] shape;
    private final double[] data;

    public Tensor(int... shape) {
        this.shape = shape.clone();
        int n = 1;
        for (int s : shape) {
            if (s <= 0) throw new IllegalArgumentException("Dimensions must be positive: " + Arrays.toString(shape));
            n *= s;
        }
        this.data = new double[n];
    }

    private Tensor(int[] shape, double[] data) {
        this.shape = shape.clone();
        this.data = data;
    }

    public static Tensor of(double[][] a) {
        int r = a.length, c = a[0].length;
        Tensor t = new Tensor(r, c);
        for (int i = 0; i < r; i++) {
            if (a[i].length != c) throw new IllegalArgumentException("Ragged array");
            System.arraycopy(a[i], 0, t.data, i * c, c);
        }
        return t;
    }

    public static Tensor identity(int n) {
        Tensor t = new Tensor(n, n);
        for (int i = 0; i < n; i++) t.data[i * n + i] = 1.0;
        return t;
    }

    public static Tensor randn(Random rnd, double scale, int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++) t.data[i] = rnd.nextGaussian() * scale;
        return t;
    }

    public int rank() { return shape.length; }
    public int[] shape() { return shape.clone(); }
    public int size() { return data.length; }
    /** Live backing array (row-major). */
    public double[] data() { return data; }
    public int rows() { requireRank2(); return shape[0]; }
    public int cols() { requireRank2(); return shape[1]; }
    public double get(int i, int j) { return data[i * shape[1] + j]; }
    public void set(int i, int j, double v) { data[i * shape[1] + j] = v; }
    public Tensor copy() { return new Tensor(shape, data.clone()); }

    public Tensor reshape(int... newShape) {
        int n = 1;
        for (int s : newShape) n *= s;
        if (n != data.length) throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
        return new Tensor(newShape, data.clone());
    }

    private void requireRank2() {
        if (shape.length != 2) throw new IllegalStateException("Rank-2 tensor required, got rank " + shape.length);
    }

    private void requireSame(Tensor o) {
        if (!Arrays.equals(shape, o.shape))
            throw new IllegalArgumentException("Shape mismatch: " + Arrays.toString(shape) + " vs " + Arrays.toString(o.shape));
    }

    public Tensor map(DoubleUnaryOperator f) {
        Tensor r = new Tensor(shape);
        for (int i = 0; i < data.length; i++) r.data[i] = f.applyAsDouble(data[i]);
        return r;
    }

    public Tensor add(Tensor o) {
        requireSame(o);
        Tensor r = new Tensor(shape);
        for (int i = 0; i < data.length; i++) r.data[i] = data[i] + o.data[i];
        return r;
    }

    public Tensor sub(Tensor o) {
        requireSame(o);
        Tensor r = new Tensor(shape);
        for (int i = 0; i < data.length; i++) r.data[i] = data[i] - o.data[i];
        return r;
    }

    /** Elementwise product. */
    public Tensor mul(Tensor o) {
        requireSame(o);
        Tensor r = new Tensor(shape);
        for (int i = 0; i < data.length; i++) r.data[i] = data[i] * o.data[i];
        return r;
    }

    public Tensor scale(double s) {
        Tensor r = new Tensor(shape);
        for (int i = 0; i < data.length; i++) r.data[i] = data[i] * s;
        return r;
    }

    /** This (n x d) plus a (1 x d) row, broadcast over rows. */
    public Tensor addRow(Tensor row) {
        requireRank2();
        row.requireRank2();
        int n = shape[0], d = shape[1];
        if (row.shape[0] != 1 || row.shape[1] != d) throw new IllegalArgumentException("Row must be 1 x " + d);
        Tensor r = new Tensor(n, d);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) r.data[i * d + j] = data[i * d + j] + row.data[j];
        return r;
    }

    /** Column sums as a 1 x d tensor. */
    public Tensor sumRows() {
        requireRank2();
        int n = shape[0], d = shape[1];
        Tensor r = new Tensor(1, d);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) r.data[j] += data[i * d + j];
        return r;
    }

    public Tensor matmul(Tensor o) {
        requireRank2();
        o.requireRank2();
        int n = shape[0], k = shape[1], m = o.shape[1];
        if (k != o.shape[0]) throw new IllegalArgumentException("Inner dimensions differ: " + k + " vs " + o.shape[0]);
        Tensor r = new Tensor(n, m);
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < k; t++) {
                double a = data[i * k + t];
                if (a == 0.0) continue;
                for (int j = 0; j < m; j++) r.data[i * m + j] += a * o.data[t * m + j];
            }
        }
        return r;
    }

    public Tensor transpose() {
        requireRank2();
        int n = shape[0], m = shape[1];
        Tensor r = new Tensor(m, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) r.data[j * n + i] = data[i * m + j];
        return r;
    }

    public double sum() {
        double s = 0;
        for (double v : data) s += v;
        return s;
    }

    public double frobenius() {
        double s = 0;
        for (double v : data) s += v * v;
        return Math.sqrt(s);
    }

    public double maxAbs() {
        double m = 0;
        for (double v : data) m = Math.max(m, Math.abs(v));
        return m;
    }

    public int argmaxRow(int i) {
        int d = cols(), best = 0;
        for (int j = 1; j < d; j++) if (get(i, j) > get(i, best)) best = j;
        return best;
    }

    @Override
    public String toString() {
        return "Tensor" + Arrays.toString(shape) + (data.length <= 12 ? Arrays.toString(data) : "[...]");
    }
}
