"""Milestone 2 -- Python side of the ONNX integration proof.

Builds a tiny convolutional classifier, exports it to ONNX, runs it with ONNX Runtime (Python),
and writes the files the Java smoke test needs to check that Java produces the same numbers.

Output files (in --out-dir, default build/smoke):
    model.onnx            the exported model (dynamic batch axis)
    input.f32             raw little-endian float32 input batch, shape from meta
    expected_output.f32   reference output (from PyTorch, or an independent NumPy implementation)
    meta.properties       key=value description of the above + numerical tolerances

Two builders:
    torch  (default if PyTorch is installed) -- the real deployment path: torch.onnx.export.
    onnx                                     -- constructs the same graph with onnx.helper; needs no
                                                PyTorch. The reference output is an independent NumPy
                                                implementation of the same maths.

Tolerance (documented, enforced in Python AND Java):   |actual - expected| <= ATOL + RTOL * |expected|
with ATOL = 1e-5, RTOL = 1e-4. For float32 models this small, differences are normally ~1e-7.

Run:  python -m ml.smoke.export_smoke_model --out-dir build/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ATOL = 1e-5
RTOL = 1e-4
BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 16, 16
CONV_OUT, NUM_CLASSES = 4, 3
INPUT_NAME, OUTPUT_NAME = "input", "logits"
OPSET = 17


def make_weights(seed: int) -> dict:
    """Deterministic float32 weights shared by both builders."""
    rng = np.random.RandomState(seed)
    return {
        "conv_w": rng.randn(CONV_OUT, CHANNELS, 3, 3).astype(np.float32) * 0.3,
        "conv_b": rng.randn(CONV_OUT).astype(np.float32) * 0.1,
        "fc_w": rng.randn(NUM_CLASSES, CONV_OUT).astype(np.float32) * 0.5,
        "fc_b": rng.randn(NUM_CLASSES).astype(np.float32) * 0.1,
    }


def make_input(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + 1000)
    return rng.rand(BATCH, CHANNELS, HEIGHT, WIDTH).astype(np.float32)


def numpy_reference(x: np.ndarray, w: dict) -> np.ndarray:
    """Independent float64 implementation: conv3x3(pad 1) -> ReLU -> global avg pool -> linear."""
    x64 = x.astype(np.float64)
    n, c, h, wd = x64.shape
    padded = np.pad(x64, ((0, 0), (0, 0), (1, 1), (1, 1)))
    conv = np.zeros((n, CONV_OUT, h, wd), dtype=np.float64)
    cw = w["conv_w"].astype(np.float64)
    for o in range(CONV_OUT):
        for u in range(3):
            for v in range(3):
                patch = padded[:, :, u:u + h, v:v + wd]            # (n, c, h, wd)
                conv[:, o] += np.einsum("nchw,c->nhw", patch, cw[o, :, u, v])
        conv[:, o] += float(w["conv_b"][o])
    act = np.maximum(conv, 0.0)
    pooled = act.mean(axis=(2, 3))                                  # (n, CONV_OUT)
    logits = pooled @ w["fc_w"].astype(np.float64).T + w["fc_b"].astype(np.float64)
    return logits.astype(np.float32)


def build_with_onnx_helper(path: Path, w: dict) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inits = [numpy_helper.from_array(arr, name) for name, arr in w.items()]
    nodes = [
        helper.make_node("Conv", [INPUT_NAME, "conv_w", "conv_b"], ["conv_out"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], [OUTPUT_NAME], alpha=1.0, beta=1.0, transB=1),
    ]
    graph = helper.make_graph(
        nodes, "smoke_cnn",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, ["batch", CHANNELS, HEIGHT, WIDTH])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch", NUM_CLASSES])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_with_torch(path: Path, w: dict, x: np.ndarray) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(CHANNELS, CONV_OUT, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(CONV_OUT, NUM_CLASSES),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(w["conv_w"]))
        model[0].bias.copy_(torch.from_numpy(w["conv_b"]))
        model[4].weight.copy_(torch.from_numpy(w["fc_w"]))
        model[4].bias.copy_(torch.from_numpy(w["fc_b"]))
    model.eval()
    dummy = torch.from_numpy(x)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET,
    )
    with torch.no_grad():
        return model(dummy).numpy().astype(np.float32)


def within_tolerance(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, float]:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    ok = bool(np.all(diff <= ATOL + RTOL * np.abs(expected.astype(np.float64))))
    return ok, float(diff.max())


def export_smoke(out_dir: Path, builder: str = "auto", seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if builder == "auto":
        try:
            import torch  # noqa: F401
            builder = "torch"
        except ImportError:
            builder = "onnx"

    w = make_weights(seed)
    x = make_input(seed)
    model_path = out_dir / "model.onnx"
    numpy_ref = numpy_reference(x, w)

    if builder == "torch":
        expected = build_with_torch(model_path, w, x)
        ok_ref, d_ref = within_tolerance(expected, numpy_ref)
        if not ok_ref:
            raise SystemExit(f"FAIL: torch output disagrees with NumPy reference (max diff {d_ref:.3e})")
    elif builder == "onnx":
        build_with_onnx_helper(model_path, w)
        expected = numpy_ref
    else:
        raise SystemExit(f"unknown builder {builder!r}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    py_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    ok_full, d_full = within_tolerance(py_out, expected)
    py_out_b1 = sess.run([OUTPUT_NAME], {INPUT_NAME: x[:1]})[0]       # dynamic batch axis check
    ok_b1, d_b1 = within_tolerance(py_out_b1, expected[:1])
    if not (ok_full and ok_b1):
        raise SystemExit(f"FAIL: Python ONNX Runtime disagrees with reference (batch2 {d_full:.3e}, batch1 {d_b1:.3e})")

    x.astype("<f4").tofile(out_dir / "input.f32")
    expected.astype("<f4").tofile(out_dir / "expected_output.f32")
    meta = {
        "builder": builder,
        "seed": seed,
        "onnx_opset": OPSET,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_shape": ",".join(str(d) for d in x.shape),
        "output_shape": ",".join(str(d) for d in expected.shape),
        "atol": ATOL,
        "rtol": RTOL,
        "python_ort_version": ort.__version__,
        "python_max_abs_diff_batch2": f"{d_full:.3e}",
        "python_max_abs_diff_batch1": f"{d_b1:.3e}",
    }
    (out_dir / "meta.properties").write_text("".join(f"{k}={v}\n" for k, v in meta.items()))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="build/smoke")
    ap.add_argument("--builder", choices=["auto", "torch", "onnx"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    meta = export_smoke(Path(args.out_dir), args.builder, args.seed)
    print(f"[python] builder={meta['builder']}  onnxruntime={meta['python_ort_version']}")
    print(f"[python] ORT(Python) vs reference: max|diff| batch2={meta['python_max_abs_diff_batch2']} "
          f"batch1={meta['python_max_abs_diff_batch1']}  (tolerance atol={ATOL}, rtol={RTOL})")
    print(f"[python] wrote {Path(args.out_dir) / 'model.onnx'}")
    print("[python] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
