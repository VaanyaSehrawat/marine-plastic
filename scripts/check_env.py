"""Verify the Python environment. Usage: python scripts/check_env.py [--minimal]

Prints versions of every required library and exits non-zero if something required is missing.
"""
import importlib
import platform
import sys

minimal = "--minimal" in sys.argv
required = ["numpy", "PIL", "yaml", "tqdm", "pytest", "onnx", "onnxruntime"]
full = ["torch", "torchvision", "scipy", "matplotlib", "pandas", "pycocotools", "imagehash"]

print(f"python      {sys.version.split()[0]}  ({sys.executable})")
print(f"machine     {platform.machine()}  system={platform.system()} {platform.release()}")
bad = []
for name in required + ([] if minimal else full):
    try:
        mod = importlib.import_module(name)
        print(f"{name:<12}{getattr(mod, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"{name:<12}MISSING ({exc})")
        bad.append(name)

if "onnxruntime" not in bad:
    import onnxruntime as ort
    print(f"ort providers: {ort.get_available_providers()}")
if not minimal and "torch" not in bad:
    import torch
    mps = getattr(torch.backends, "mps", None)
    print(f"torch MPS (Apple GPU) available: {bool(mps and mps.is_available())}  (informational; training uses Colab)")
if platform.system() == "Darwin" and platform.machine() != "arm64":
    print("WARNING: Python is not running natively on arm64 (Rosetta?)")
    bad.append("arch")

if bad:
    print("FAILED: " + ", ".join(bad))
    sys.exit(1)
print("Python environment OK")
