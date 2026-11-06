"""Static guards: pins stay consistent and the Java smoke test can only ever use the real ONNX Runtime."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_value(key):
    text = (ROOT / "config" / "toolchain.env").read_text()
    m = re.search(rf'^{key}="(?:\$\{{{key}:-)?([^"}}]*)', text, re.M)
    return m.group(1) if m else None


def test_ort_python_and_java_versions_match():
    java_v = _env_value("ORT_VERSION")
    req = (ROOT / "requirements" / "base.txt").read_text()
    assert f"onnxruntime=={java_v}" in req
    assert java_v == "1.20.0"


def test_jar_sha256_is_pinned():
    sha = _env_value("ORT_JAR_SHA256")
    assert sha and re.fullmatch(r"[0-9a-f]{64}", sha)


def test_no_ort_classes_or_stubs_shadow_the_real_api():
    java = ROOT / "java"
    assert not list(java.rglob("ai/onnxruntime"))
    for f in java.rglob("*.java"):
        text = f.read_text()
        assert "package ai.onnxruntime" not in text
        assert "stub" not in text.lower() and "fake" not in text.lower()


def test_smoke_test_imports_real_ort_api():
    src = (ROOT / "java/app/src/main/java/mp/app/smoke/OnnxSmokeTest.java").read_text()
    assert "import ai.onnxruntime.OrtEnvironment;" in src and "createSession" in src


def test_every_requirement_is_pinned_exactly():
    for f in (ROOT / "requirements").glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-r")):
                assert "==" in line, f"{f.name}: unpinned requirement {line}"
