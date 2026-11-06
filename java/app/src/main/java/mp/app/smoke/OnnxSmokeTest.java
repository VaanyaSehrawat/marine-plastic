package mp.app.smoke;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

/**
 * Milestone 2 -- Java side of the ONNX integration proof.
 *
 * Loads build/smoke/model.onnx with ONNX Runtime Java, runs the same input that Python used, and checks
 * the result against the reference output within the documented tolerance
 * (|actual - expected| <= atol + rtol * |expected|). Also checks a different batch size (dynamic axis).
 *
 * Usage: java -cp "java/out:java/lib/*" mp.app.smoke.OnnxSmokeTest [modelDir] [--expect-arch aarch64]
 * Exit code 0 = PASS, 1 = numerical mismatch, 2 = setup/IO/architecture problem.
 */
public final class OnnxSmokeTest {

    private OnnxSmokeTest() { }

    public static void main(String[] args) {
        int code;
        try {
            code = run(args);
        } catch (Throwable t) {
            System.out.println("[java] ERROR: " + t);
            t.printStackTrace(System.out);
            code = 2;
        }
        System.exit(code);
    }

    static int run(String[] args) throws IOException, OrtException {
        String dirArg = "build/smoke";
        String expectArch = null;
        for (int i = 0; i < args.length; i++) {
            if ("--expect-arch".equals(args[i]) && i + 1 < args.length) {
                expectArch = args[++i];
            } else {
                dirArg = args[i];
            }
        }
        Path dir = Path.of(dirArg);

        String osArch = System.getProperty("os.arch");
        System.out.printf("[java] java.version=%s  vm=%s  os=%s  os.arch=%s%n",
                System.getProperty("java.version"), System.getProperty("java.vm.name"),
                System.getProperty("os.name"), osArch);
        if (expectArch != null && !expectArch.equals(osArch)) {
            System.out.println("[java] FAIL: JVM architecture is '" + osArch + "' but '" + expectArch
                    + "' was expected. An x86_64 JVM (Rosetta) would load the wrong native libraries.");
            return 2;
        }

        Properties meta = new Properties();
        try (InputStream in = Files.newInputStream(dir.resolve("meta.properties"))) {
            meta.load(in);
        }
        String inputName = meta.getProperty("input_name");
        long[] inShape = parseShape(meta.getProperty("input_shape"));
        long[] outShape = parseShape(meta.getProperty("output_shape"));
        double atol = Double.parseDouble(meta.getProperty("atol"));
        double rtol = Double.parseDouble(meta.getProperty("rtol"));

        float[] input = readFloats(dir.resolve("input.f32"));
        float[] expected = readFloats(dir.resolve("expected_output.f32"));
        long batch = inShape[0];
        int perIn = (int) (input.length / batch);
        int perOut = (int) (expected.length / batch);
        if (input.length != product(inShape) || expected.length != product(outShape)) {
            System.out.println("[java] FAIL: file sizes do not match shapes in meta.properties");
            return 2;
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        double worst = 0.0;
        boolean ok = true;
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
             OrtSession session = env.createSession(dir.resolve("model.onnx").toString(), opts)) {
            System.out.println("[java] model inputs=" + session.getInputNames());

            // Run 1: full batch.
            float[] out2 = infer(env, session, inputName, input, inShape);
            double d2 = maxViolation(out2, expected, atol, rtol);
            // Run 2: batch of one (dynamic batch axis).
            long[] shape1 = inShape.clone();
            shape1[0] = 1;
            float[] out1 = infer(env, session, inputName, slice(input, 0, perIn), shape1);
            double d1 = maxViolation(out1, slice(expected, 0, perOut), atol, rtol);
            System.out.printf(Locale.ROOT, "[java] max|diff| batch=%d: %.3e   batch=1: %.3e   (atol=%s, rtol=%s)%n",
                    batch, maxAbsDiff(out2, expected), maxAbsDiff(out1, slice(expected, 0, perOut)), atol, rtol);
            ok = d2 <= 0.0 && d1 <= 0.0;
            worst = Math.max(maxAbsDiff(out2, expected), maxAbsDiff(out1, slice(expected, 0, perOut)));

            // Informational latency (not part of PASS/FAIL).
            for (int i = 0; i < 5; i++) {
                infer(env, session, inputName, input, inShape);
            }
            int runs = 50;
            long t0 = System.nanoTime();
            for (int i = 0; i < runs; i++) {
                infer(env, session, inputName, input, inShape);
            }
            double ms = (System.nanoTime() - t0) / 1e6 / runs;
            System.out.printf(Locale.ROOT, "[java] mean latency (batch=%d, tiny model, CPU): %.3f ms/run (informational)%n", batch, ms);
        }

        if (ok) {
            System.out.printf(Locale.ROOT, "[java] PASS (worst abs diff %.3e)%n", worst);
            return 0;
        }
        System.out.println("[java] FAIL: Java output differs from reference beyond tolerance");
        return 1;
    }

    private static float[] infer(OrtEnvironment env, OrtSession session, String inputName,
                                 float[] data, long[] shape) throws OrtException {
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * Float.BYTES).order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(data);
        fb.rewind();
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, fb, shape);
             OrtSession.Result result = session.run(Map.of(inputName, tensor))) {
            OnnxValue value = result.get(0);
            FloatBuffer out = ((OnnxTensor) value).getFloatBuffer();
            float[] arr = new float[out.remaining()];
            out.get(arr);
            return arr;
        }
    }

    /** Returns the largest amount by which any element exceeds the tolerance band (0 if all inside). */
    private static double maxViolation(float[] actual, float[] expected, double atol, double rtol) {
        if (actual.length != expected.length) {
            return Double.POSITIVE_INFINITY;
        }
        double worst = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double diff = Math.abs((double) actual[i] - (double) expected[i]);
            double allowed = atol + rtol * Math.abs((double) expected[i]);
            worst = Math.max(worst, diff - allowed);
        }
        return Math.max(worst, 0.0);
    }

    private static double maxAbsDiff(float[] a, float[] b) {
        if (a.length != b.length) {
            return Double.POSITIVE_INFINITY;
        }
        double m = 0.0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs((double) a[i] - (double) b[i]));
        }
        return m;
    }

    private static float[] slice(float[] src, int from, int len) {
        float[] r = new float[len];
        System.arraycopy(src, from, r, 0, len);
        return r;
    }

    private static float[] readFloats(Path p) throws IOException {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
        float[] r = new float[bb.remaining() / Float.BYTES];
        bb.asFloatBuffer().get(r);
        return r;
    }

    private static long[] parseShape(String s) {
        String[] parts = s.trim().split(",");
        long[] r = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            r[i] = Long.parseLong(parts[i].trim());
        }
        return r;
    }

    private static long product(long[] shape) {
        long p = 1;
        for (long d : shape) {
            p *= d;
        }
        return p;
    }
}
