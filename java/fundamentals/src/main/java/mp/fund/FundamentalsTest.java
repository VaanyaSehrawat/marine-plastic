package mp.fund;

import java.util.Random;

public final class FundamentalsTest {
    private static int checks = 0;

    private static void check(boolean ok, String msg) {
        checks++;
        if (!ok) throw new AssertionError("FUNDAMENTALS TEST FAILED: " + msg);
    }

    private static void near(double got, double want, double tol, String msg) {
        check(Math.abs(got - want) <= tol, msg + " (got " + got + ", expected " + want + ")");
    }

    public static void main(String[] args) {
        tensorTests();
        svdTests();
        optimizerTests();
        mlpTests();
        hebbianTest();
        competitiveTest();
        System.out.println("FUNDAMENTALS TEST: PASS (" + checks + " checks)");
    }

    private static void tensorTests() {
        Tensor a = Tensor.of(new double[][] {{1, 2, 3}, {4, 5, 6}});
        Tensor b = Tensor.of(new double[][] {{7, 8}, {9, 10}, {11, 12}});
        Tensor c = a.matmul(b);
        near(c.get(0, 0), 58, 1e-12, "matmul [0,0]");
        near(c.get(0, 1), 64, 1e-12, "matmul [0,1]");
        near(c.get(1, 0), 139, 1e-12, "matmul [1,0]");
        near(c.get(1, 1), 154, 1e-12, "matmul [1,1]");
        Tensor t = a.transpose();
        check(t.rows() == 3 && t.cols() == 2 && t.get(2, 1) == 6.0, "transpose");
        Tensor r = a.addRow(Tensor.of(new double[][] {{10, 20, 30}}));
        near(r.get(1, 2), 36, 1e-12, "addRow");
        near(a.sumRows().get(0, 1), 7, 1e-12, "sumRows");
        near(a.sub(a).maxAbs(), 0, 0, "sub");
        System.out.println("[tensor] ok");
    }

    private static void svdTests() {
        Svd d = Svd.decompose(Tensor.of(new double[][] {{3, 0}, {0, 4}}));
        near(d.s()[0], 4.0, 1e-12, "singular value 1");
        near(d.s()[1], 3.0, 1e-12, "singular value 2");
        Random rnd = new Random(7);
        int[][] shapes = {{6, 4}, {4, 7}, {40, 25}, {25, 40}, {5, 5}};
        double worstRec = 0, worstOrth = 0;
        for (int[] sh : shapes) {
            Tensor a = Tensor.randn(rnd, 1.0, sh[0], sh[1]);
            Svd s = Svd.decompose(a);
            int r = Math.min(sh[0], sh[1]);
            double[] sv = s.s();
            String tag = sh[0] + "x" + sh[1];
            check(sv.length == r, "number of singular values " + tag);
            double rec = a.sub(s.reconstruct(r)).frobenius();
            double ou = s.u().transpose().matmul(s.u()).sub(Tensor.identity(r)).maxAbs();
            double ov = s.v().transpose().matmul(s.v()).sub(Tensor.identity(r)).maxAbs();
            worstRec = Math.max(worstRec, rec);
            worstOrth = Math.max(worstOrth, Math.max(ou, ov));
            check(rec < 1e-9, "reconstruction error " + tag + " = " + rec);
            check(ou < 1e-9 && ov < 1e-9, "orthogonality " + tag);
            for (int i = 1; i < r; i++) check(sv[i - 1] >= sv[i] && sv[i] >= 0, "descending order " + tag);
            int k = 2;
            double tail = 0;
            for (int i = k; i < r; i++) tail += sv[i] * sv[i];
            near(a.sub(s.reconstruct(k)).frobenius(), Math.sqrt(tail), 1e-8, "Eckart-Young " + tag);
        }
        System.out.printf("[svd] ok | worst reconstruction error %.2e, worst orthogonality error %.2e%n", worstRec, worstOrth);
    }

    private static double[] minimize(Optimizer o, int steps) {
        double[] w = {0.0, 10.0, -5.0};
        for (int t = 0; t < steps; t++) {
            double[] g = new double[3];
            for (int i = 0; i < 3; i++) g[i] = 2.0 * (w[i] - 3.0);
            o.update(0, w, g);
        }
        return w;
    }

    private static double dist3(double[] w) {
        double m = 0;
        for (double v : w) m = Math.max(m, Math.abs(v - 3.0));
        return m;
    }

    private static void optimizerTests() {
        double e1 = dist3(minimize(new Optimizer.Sgd(0.1), 200));
        double e2 = dist3(minimize(new Optimizer.Momentum(0.05, 0.9), 400));
        double e3 = dist3(minimize(new Optimizer.Adam(0.05), 3000));
        check(e1 < 1e-3, "SGD converges, error " + e1);
        check(e2 < 1e-3, "Momentum converges, error " + e2);
        check(e3 < 0.1, "Adam converges, error " + e3);
        System.out.printf("[optimizers] ok | final error SGD %.2e, Momentum %.2e, Adam %.2e%n", e1, e2, e3);
    }

    private static Tensor xorX() { return Tensor.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}}); }
    private static final int[] XOR_Y = {0, 1, 1, 0};

    private static int stepsToLoss(Optimizer opt, double target, int cap) {
        Mlp m = new Mlp(new int[] {2, 8, 2}, Mlp.Activation.TANH, 5);
        Tensor x = xorX();
        for (int s = 1; s <= cap; s++) {
            double loss = m.trainStep(x, XOR_Y, opt);
            if (loss < target) return s;
        }
        return -1;
    }

    private static void mlpTests() {
        Mlp m = new Mlp(new int[] {3, 5, 4, 2}, Mlp.Activation.TANH, 11);
        Random rnd = new Random(3);
        Tensor x = Tensor.randn(rnd, 1.0, 6, 3);
        int[] y = {0, 1, 1, 0, 1, 0};
        Tensor[] gw = new Tensor[m.layers()], gb = new Tensor[m.layers()];
        m.computeGradients(x, y, gw, gb);
        double h = 1e-5, worst = 0;
        int bad = 0, total = 0;
        for (int l = 0; l < m.layers(); l++) {
            Tensor[] params = {m.weight(l), m.bias(l)};
            Tensor[] grads = {gw[l], gb[l]};
            for (int pi = 0; pi < 2; pi++) {
                double[] arr = params[pi].data();
                for (int idx = 0; idx < arr.length; idx++) {
                    double old = arr[idx];
                    arr[idx] = old + h;
                    double lp = m.loss(x, y);
                    arr[idx] = old - h;
                    double lm = m.loss(x, y);
                    arr[idx] = old;
                    double num = (lp - lm) / (2 * h), ana = grads[pi].data()[idx];
                    worst = Math.max(worst, Math.abs(num - ana));
                    total++;
                    if (Math.abs(num - ana) > 1e-7 + 1e-5 * Math.max(Math.abs(num), Math.abs(ana))) bad++;
                }
            }
        }
        check(bad == 0, "gradient check failed for " + bad + " of " + total + " parameters");
        System.out.printf("[mlp gradient check] ok | %d parameters, worst absolute difference %.2e%n", total, worst);

        Mlp xor = new Mlp(new int[] {2, 8, 2}, Mlp.Activation.TANH, 5);
        Optimizer.Adam adam = new Optimizer.Adam(0.05);
        Tensor xx = xorX();
        double loss = 1;
        int steps = 0;
        while (steps < 3000 && loss > 0.01) { loss = xor.trainStep(xx, XOR_Y, adam); steps++; }
        check(xor.accuracy(xx, XOR_Y) == 1.0, "XOR accuracy with Adam");
        check(loss < 0.05, "XOR loss with Adam = " + loss);
        System.out.printf("[mlp xor] ok | accuracy 100%% after %d Adam steps, loss %.4f%n", steps, loss);

        int sSgd = stepsToLoss(new Optimizer.Sgd(0.5), 0.05, 5000);
        int sMom = stepsToLoss(new Optimizer.Momentum(0.1, 0.9), 0.05, 5000);
        int sAdam = stepsToLoss(new Optimizer.Adam(0.05), 0.05, 5000);
        check(sAdam > 0, "Adam reaches XOR loss 0.05 within 5000 steps");
        System.out.println("[optimizer comparison on XOR, steps to loss < 0.05] SGD(lr 0.5): " + (sSgd < 0 ? "not reached in 5000" : sSgd)
            + " | Momentum(lr 0.1, beta 0.9): " + (sMom < 0 ? "not reached in 5000" : sMom)
            + " | Adam(lr 0.05): " + sAdam);
    }

    private static void hebbianTest() {
        Random rnd = new Random(1);
        int n = 2000;
        double th = Math.toRadians(35);
        double[][] pts = new double[n][2];
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) {
            double a = 3.0 * rnd.nextGaussian(), b = 0.5 * rnd.nextGaussian();
            pts[i][0] = a * Math.cos(th) - b * Math.sin(th);
            pts[i][1] = a * Math.sin(th) + b * Math.cos(th);
            mx += pts[i][0]; my += pts[i][1];
        }
        mx /= n; my /= n;
        for (double[] p : pts) { p[0] -= mx; p[1] -= my; }
        Tensor x = Tensor.of(pts);
        double[] w = Hebbian.oja(x, 0.002, 30, 9);
        Svd s = Svd.decompose(x);
        double v0 = s.v().get(0, 0), v1 = s.v().get(1, 0);
        double nw = Math.hypot(w[0], w[1]);
        double cos = Math.abs(w[0] * v0 + w[1] * v1) / (nw * Math.hypot(v0, v1));
        check(cos > 0.99, "Oja direction vs first singular vector, cosine " + cos);
        near(nw, 1.0, 0.05, "Oja weight norm");
        System.out.printf("[hebbian] ok | |cos(Oja weight, SVD first right singular vector)| = %.5f, weight norm %.4f%n", cos, nw);
    }

    private static void competitiveTest() {
        Random rnd = new Random(21);
        double[][] centers = {{0, 0}, {10, 0}, {5, 9}};
        int per = 100;
        double[][] pts = new double[3 * per][2];
        int[] lab = new int[3 * per];
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < per; i++) {
                int idx = c * per + i;
                pts[idx][0] = centers[c][0] + 0.5 * rnd.nextGaussian();
                pts[idx][1] = centers[c][1] + 0.5 * rnd.nextGaussian();
                lab[idx] = c;
            }
        }
        Competitive cl = Competitive.train(Tensor.of(pts), 3, 0.3, 20, 4);
        int[] win = {-1, -1, -1};
        boolean consistent = true;
        for (int i = 0; i < pts.length; i++) {
            int w = cl.winner(pts[i]);
            if (win[lab[i]] == -1) win[lab[i]] = w; else if (win[lab[i]] != w) consistent = false;
        }
        check(consistent, "each cluster is claimed by a single prototype");
        check(win[0] != win[1] && win[0] != win[2] && win[1] != win[2], "clusters use distinct prototypes");
        double worst = 0;
        for (int c = 0; c < 3; c++) {
            double[] p = cl.prototypes()[win[c]];
            worst = Math.max(worst, Math.hypot(p[0] - centers[c][0], p[1] - centers[c][1]));
        }
        check(worst < 0.3, "prototype distance to true centre " + worst);
        System.out.printf("[competitive] ok | 300 points, 3 clusters, purity 100%%, worst prototype-to-centre distance %.3f%n", worst);
    }
}
