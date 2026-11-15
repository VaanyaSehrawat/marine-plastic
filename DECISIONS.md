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

## Update 14 Nov 2026 - D-08 (supersedes D-06 for the primary detector)
| ID | Decision | Reason / evidence |
|---|---|---|
| D-08 | Primary detector is Ultralytics YOLO11s (1280 px), trained on Kaggle T4 on the official BePLi v2 splits, exported to ONNX and run from Java with ONNX Runtime 1.20.0. Torchvision Faster R-CNN / RetinaNet are optional comparison baselines only. | Official-test mAP50 0.237, mAP50-95 0.144; class-agnostic mAP50 0.558. ONNX vs PyTorch max box difference 0.0072 px. Java fixture test PASS (scripts/detect_java.sh). Licence: Ultralytics code and weights are AGPL-3.0 and BePLi is CC BY-NC-SA 4.0; acceptable for this non-commercial academic project, and any redistribution must follow both. |

## Update 15 Nov 2026 - D-09 (detector and learned tile scoring)
| ID | Decision | Reason / evidence |
|---|---|---|
| D-09 | Dashboard default detector is YOLO11m (YOLO11s stays available with `MODEL_DIR=yolo11s_1280 bash scripts/run_dashboard.sh`). Learned tile priority uses a 71,233-parameter route-context transformer (ONNX, run in Java), trained on val-split detections, selectable in the dashboard next to the heuristic scoring. | Detectors (official splits, 1280 px, conf 0.001, NMS IoU 0.6, one training run each): YOLO11m val mAP50-95 0.182 and test 0.157; YOLO11s val 0.162 and test 0.144. Java CPU inference on the student's Mac (3 fixture images, after one warm-up): YOLO11m 1835-1879 ms, YOLO11s 549-694 ms. Tile ranking on test (610 images with at least 5 ground-truth tiles): Spearman transformer 0.661 +- 0.001, tile MLP 0.660 +- 0.001, detector-only score 0.531, random -0.007; top-5 overlap 0.591, 0.590, 0.524, 0.095. Cross-tile context gave no measurable gain over the MLP. Labels are ground-truth boxes weighted by the same class weights, so this measures agreement with a proxy, not cleanup efficiency. Java checks: scripts/test_route.sh and scripts/run_dashboard.sh --selftest PASS. |
