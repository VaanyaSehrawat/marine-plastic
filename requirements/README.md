# Requirements

| File | Used by | Contents |
|---|---|---|
| `base.txt` | `bash setup.sh --minimal`, tests | numpy, Pillow, PyYAML, tqdm, pytest, onnx, onnxruntime |
| `train.txt` | `bash setup.sh` (default) | base + torch, torchvision, scipy, matplotlib, pandas, pycocotools, imagehash |

Versions are exact pins. Wheels for every pin were confirmed to exist for CPython 3.12 on macOS arm64
(`pip download --platform macosx_14_0_arm64`, and a full `uv pip compile` for macOS arm64) on 2026-11-03/05. They were not yet *installed* on the Mac.
`onnxruntime` (Python) and `ORT_VERSION` in `config/toolchain.env` (Java) must stay on the same version.
On Colab, keep Colab's preinstalled torch and run `pip install onnx==1.17.0 onnxruntime==1.20.0 pycocotools imagehash`.
