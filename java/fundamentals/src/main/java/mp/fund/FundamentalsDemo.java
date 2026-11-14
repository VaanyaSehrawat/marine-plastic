package mp.fund;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/** SVD image compression demo. Usage: FundamentalsDemo input.png output.png */
public final class FundamentalsDemo {
    public static void main(String[] a) throws Exception {
        System.setProperty("java.awt.headless", "true");
        Path in = Path.of(a[0]), out = Path.of(a[1]);
        BufferedImage src = ImageIO.read(in.toFile());
        int W = 320, H = 240;
        BufferedImage gray = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, W, H, null);
        g.dispose();

        Tensor m = new Tensor(H, W);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) m.set(y, x, gray.getRaster().getSample(x, y, 0));

        long t0 = System.nanoTime();
        Svd svd = Svd.decompose(m);
        double ms = (System.nanoTime() - t0) / 1e6;
        double[] s = svd.s();
        double total = 0;
        for (double v : s) total += v * v;
        double norm = m.frobenius();
        System.out.printf("SVD of %dx%d image computed in %.0f ms; largest singular value %.1f%n", H, W, ms, s[0]);

        BufferedImage montage = new BufferedImage(W * 4, H, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) montage.getRaster().setSample(x, y, 0, gray.getRaster().getSample(x, y, 0));

        int[] ks = {5, 20, 60};
        for (int i = 0; i < ks.length; i++) {
            int k = ks[i];
            Tensor rk = svd.reconstruct(k);
            double err = m.sub(rk).frobenius() / norm;
            double kept = 0;
            for (int t = 0; t < k; t++) kept += s[t] * s[t];
            double ratio = k * (H + W + 1.0) / ((double) H * W);
            System.out.printf("rank %2d: relative error %.4f | energy kept %.2f%% | stored numbers %.1f%% of original%n",
                k, err, 100.0 * kept / total, 100.0 * ratio);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    int v = (int) Math.round(Math.max(0, Math.min(255, rk.get(y, x))));
                    montage.getRaster().setSample((i + 1) * W + x, y, 0, v);
                }
        }
        Files.createDirectories(out.getParent());
        ImageIO.write(montage, "png", out.toFile());
        System.out.println("Saved " + out + " (left to right: original, rank 5, rank 20, rank 60)");
    }
}
