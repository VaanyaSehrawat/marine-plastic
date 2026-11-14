package mp.fund;

import java.util.Arrays;

/** Singular value decomposition A = U diag(s) V^T by one-sided Jacobi (Hestenes). Handles tall and wide matrices. */
public final class Svd {
    private final Tensor u, v;
    private final double[] s;

    private Svd(Tensor u, double[] s, Tensor v) {
        this.u = u;
        this.s = s;
        this.v = v;
    }

    /** m x r, r = min(m, n). */
    public Tensor u() { return u; }
    /** Singular values, descending. */
    public double[] s() { return s.clone(); }
    /** n x r. */
    public Tensor v() { return v; }

    public static Svd decompose(Tensor a) {
        int m0 = a.rows(), n0 = a.cols();
        boolean wide = m0 < n0;
        int m = wide ? n0 : m0, n = wide ? m0 : n0; // work on the tall matrix (m x n)
        double[][] uc = new double[n][m];            // columns of the working matrix
        for (int i = 0; i < m0; i++) {
            for (int j = 0; j < n0; j++) {
                double x = a.get(i, j);
                if (wide) uc[i][j] = x; else uc[j][i] = x;
            }
        }
        double[][] vc = new double[n][n];
        for (int i = 0; i < n; i++) vc[i][i] = 1.0;

        for (int sweep = 0; sweep < 100; sweep++) {
            boolean rotated = false;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double[] up = uc[p], uq = uc[q];
                    double alpha = 0, beta = 0, gamma = 0;
                    for (int i = 0; i < m; i++) {
                        alpha += up[i] * up[i];
                        beta += uq[i] * uq[i];
                        gamma += up[i] * uq[i];
                    }
                    if (gamma == 0.0 || Math.abs(gamma) <= 1e-14 * Math.sqrt(alpha * beta)) continue;
                    rotated = true;
                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t = (zeta >= 0 ? 1.0 : -1.0) / (Math.abs(zeta) + Math.sqrt(1.0 + zeta * zeta));
                    double c = 1.0 / Math.sqrt(1.0 + t * t), sn = c * t;
                    rotate(up, uq, c, sn);
                    rotate(vc[p], vc[q], c, sn);
                }
            }
            if (!rotated) break;
        }

        double[] sig = new double[n];
        for (int p = 0; p < n; p++) {
            double ss = 0;
            for (int i = 0; i < m; i++) ss += uc[p][i] * uc[p][i];
            sig[p] = Math.sqrt(ss);
        }
        Integer[] ord = new Integer[n];
        for (int i = 0; i < n; i++) ord[i] = i;
        Arrays.sort(ord, (x, y) -> Double.compare(sig[y], sig[x]));

        double[] sOut = new double[n];
        Tensor ut = new Tensor(m, n), vt = new Tensor(n, n);
        for (int k = 0; k < n; k++) {
            int c = ord[k];
            sOut[k] = sig[c];
            double inv = sig[c] > 1e-300 ? 1.0 / sig[c] : 0.0;
            for (int i = 0; i < m; i++) ut.set(i, k, uc[c][i] * inv);
            for (int i = 0; i < n; i++) vt.set(i, k, vc[c][i]);
        }
        return wide ? new Svd(vt, sOut, ut) : new Svd(ut, sOut, vt);
    }

    private static void rotate(double[] x, double[] y, double c, double s) {
        for (int i = 0; i < x.length; i++) {
            double a = x[i], b = y[i];
            x[i] = c * a - s * b;
            y[i] = s * a + c * b;
        }
    }

    /** Rank-k approximation, the best in Frobenius norm (Eckart-Young). */
    public Tensor reconstruct(int k) {
        int m = u.rows(), n = v.rows();
        if (k < 0 || k > s.length) throw new IllegalArgumentException("k must be in [0, " + s.length + "]");
        Tensor out = new Tensor(m, n);
        double[] o = out.data();
        for (int t = 0; t < k; t++) {
            for (int i = 0; i < m; i++) {
                double ui = u.get(i, t) * s[t];
                for (int j = 0; j < n; j++) o[i * n + j] += ui * v.get(j, t);
            }
        }
        return out;
    }
}
