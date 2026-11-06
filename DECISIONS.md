# DECISIONS.md

Status key: **APPROVED** (owner agreed; reopen only if a real test fails) | **PROVISIONAL** (working assumption, to be confirmed by evidence) | **OPEN** (genuinely undecided).
Last updated: 2026-11-05.

## Approved
| ID | Decision | Reason / evidence |
|---|---|---|
| D-01 | Hybrid architecture: Python/PyTorch trains on Colab; ONNX export; Java 21 app runs inference; pure-Java fundamentals module; Java-served web dashboard. | Approved by faculty and owner. Keeps the spec's Java requirement without forcing deep-learning training into Java. |
| D-02 | **ONNX Runtime 1.20.0** for both Python (`requirements/base.txt`) and Java (`config/toolchain.env`). | 1.20.1 does not exist for Java on Maven Central (HTTP 404 on the owner's Mac; the Maven listing has only 1.20.0 in that line). 1.20.0 verified on the owner's Mac: jar downloads and contains `native/osx-aarch64/libonnxruntime.dylib` and `libonnxruntime4j_jni.dylib`; SHA-256 `c46608681692b3693914defb42fa3119ad7cb6146870581ece247e0ed5793fe8`. Do not reopen unless the real Mac smoke test fails. |
| D-03 | The jar is SHA-256 pinned and its Apple Silicon natives are validated strictly on every run (`scripts/fetch_java_deps.sh`). | A jar of Java classes can lack native libraries (historical packaging bugs in ORT 1.11/1.12). |
| D-04 | Project-local toolchain: Temurin JDK 21.0.13+9 and a uv-managed Python 3.12 under `.tools/` and `.venv/`; system Java 17 / Python 3.13 untouched; no Maven. | Verified real macOS arm64 artifacts + SHA-256 (JDK, uv). Wheels for all pins resolve for macOS arm64 / CPython 3.12. |
| D-05 | Planner works on an **image-space tile grid**; routes are inspection orders, never GPS navigation. | No public dataset provides usable coordinates. |
| D-06 | Torchvision Faster R-CNN and RetinaNet only (no YOLO). | Permissive licensing, one framework. |
| D-07 | If direct Java inference fails, fall back to a local Python HTTP detector behind the same `Detector` interface (Option B). | Pre-agreed so a failure costs hours, not days. Not needed unless the Mac smoke test fails. |

## Provisional
| ID | Decision | What would change it |
|---|---|---|
| P-01 | **BePLi v2 is the primary dataset.** | Real inspection on Colab (M3) shows unusable annotations, missing files, or a license problem. |
| P-02 | UAVVaste is an optional **out-of-domain robustness test only**; never merged into training. | Image-data licensing cannot be confirmed (repo code is Apache-2.0; image terms unverified); single generic class. |
| P-03 | BeachLitter and SeaClear are **not used**. | BeachLitter: non-plastic semantic classes, same Yamagata monitoring source (probable overlap with BePLi), CC BY-NC-SA. SeaClear: underwater ROV imagery, 40 classes incl. animals/plants. |
| P-04 | Splits are group-aware (near-duplicate clusters + optional filename blocks), built by us unless the dataset's own split passes the audit. | The audit (`ml.data.near_duplicates`) on the official split. |
| P-05 | Tile labels: positive / negative / uncertain (rules in `ml/data/tile_labels.py`); unannotated is not automatically negative. | Manual review shows annotation incompleteness is worse/better than assumed. |
| P-06 | BePLi (CC BY-NC-SA 4.0) is acceptable for this non-commercial academic project; weights/derivatives are share-alike non-commercial. | Not legal advice; re-read the record before any redistribution. |

## Open
| ID | Question | Decide by |
|---|---|---|
| O-01 | Does consecutive file numbering in BePLi correspond to site visits (enables `--block-size`)? | M3 manual review (13 Nov latest) |
| O-02 | Which exact download link / archive layout does BePLi v2 use (not verified: structure of the 8 GB archive)? | M3, first Colab run |
| O-03 | Faculty acceptance of "VAE only" for the GAN/VAE requirement. | Before 16 Nov |
| O-04 | Segmentation (Mask R-CNN) in or out. | 16 Nov cut line |
| O-05 | Number of BePLi annotations in v2 (claimed ~118,572: UNVERIFIED; v1 paper reports 119,192). | M3 inspection output |
