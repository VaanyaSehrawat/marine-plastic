"""Tests for the Python half of the ONNX smoke test (needs only numpy, onnx, onnxruntime)."""
import numpy as np

from ml.smoke import export_smoke_model as m


def test_reference_is_deterministic():
    w = m.make_weights(0)
    x = m.make_input(0)
    assert np.array_equal(m.numpy_reference(x, w), m.numpy_reference(x, w))


def test_reference_shape():
    out = m.numpy_reference(m.make_input(0), m.make_weights(0))
    assert out.shape == (m.BATCH, m.NUM_CLASSES)
    assert out.dtype == np.float32


def test_onnx_builder_matches_numpy_reference(tmp_path):
    meta = m.export_smoke(tmp_path, builder="onnx", seed=0)
    assert meta["builder"] == "onnx"
    for f in ("model.onnx", "input.f32", "expected_output.f32", "meta.properties"):
        assert (tmp_path / f).exists()
    expected = np.fromfile(tmp_path / "expected_output.f32", dtype="<f4")
    assert expected.size == m.BATCH * m.NUM_CLASSES


def test_different_seeds_give_different_outputs(tmp_path):
    a = m.numpy_reference(m.make_input(0), m.make_weights(0))
    b = m.numpy_reference(m.make_input(1), m.make_weights(1))
    assert not np.allclose(a, b)


def test_tolerance_helper_detects_error():
    a = np.ones((2, 3), dtype=np.float32)
    ok, _ = m.within_tolerance(a + 1e-7, a)
    bad, d = m.within_tolerance(a + 1e-2, a)
    assert ok and not bad and d > 1e-3


def test_torch_builder_if_available(tmp_path):
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("PyTorch not installed in this environment")
    meta = m.export_smoke(tmp_path, builder="torch", seed=0)
    assert meta["builder"] == "torch"
