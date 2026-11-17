# Marine Plastic Detection and Coastal Cleanup Planner

Fifth-semester B.Tech CSE (AI & ML) project, ocean-conservation domain. Individual project.

Deep-learning models are trained in Python (PyTorch and Ultralytics on a free Kaggle GPU), exported to ONNX, and run inside a Java 21 application. The application detects beach litter in photographs, scores image tiles for cleanup priority, plans an inspection route over the high-priority tiles, and serves the result as a local web dashboard.

## Architecture

```
Kaggle / Colab (Python)                         Mac (Java 21)
BePLi v2 -> YOLO11s / YOLO11m  ---ONNX--->  YoloOnnxDetector (letterbox, NMS)  -> detections
detections -> route-context transformer ---ONNX---> RouteScorer -> tile priorities
                                                tiles -> CleanupPlanner (nearest neighbour + 2-opt)
                                                DashboardServer + dashboard.html (http://127.0.0.1:8080)
pure-Java fundamentals module: Tensor, SVD, MLP, SGD / Momentum / Adam, Hebbian (Oja), competitive learning
VAE and DCGAN: trained and evaluated offline in Python (no Java inference)
```

The routes are inspection orders over an image-space tile grid. They are not GPS navigation.

## Quick start (macOS, Apple Silicon)

```
bash scripts/run_dashboard.sh                           # YOLO11m detector + learned scoring, opens http://127.0.0.1:8080
MODEL_DIR=yolo11s_1280 bash scripts/run_dashboard.sh    # smaller, faster detector
bash scripts/verify_all.sh                              # runs every check, a few minutes, writes docs/VERIFICATION.md
```

Fresh machine: `bash setup.sh` installs the project-local JDK 21 and Python 3.12 (see docs/SETUP.md). The model weights are not stored in git; their SHA-256 values are in `models/*/SHA256.txt`.

## Repository layout

| path | content |
|---|---|
| `java/app/src/main/java/mp/app/detect` | ONNX Runtime YOLO detector, JSON reader, fixture test, command-line demo |
| `java/app/src/main/java/mp/app/route` | route-context transformer scorer and its test |
| `java/app/src/main/java/mp/app/planner` | tile scoring, nearest-neighbour + 2-opt routes, serpentine baseline, tests |
| `java/app/src/main/java/mp/app/server` | local HTTP dashboard server |
| `java/fundamentals/src/main/java/mp/fund` | tensor, SVD, MLP, optimizers, Hebbian and competitive learning, tests, SVD demo |
| `ml/` and `tests/python` | dataset tools and their tests |
| `notebooks/` | Kaggle and Colab notebooks that produced the models and results (outputs stripped) |
| `models/` | model metadata, hashes, test fixtures and the small route transformer (detector weights are not in git) |
| `results/`, `docs/` | measured results and the generated reports |
| `scripts/` | run, test, evaluation and verification scripts |

## Key results (full tables: docs/RESULTS.md, docs/PLANNER_EVAL.md, docs/DETECTOR_ANALYSIS.md)

- Dataset: BePLi v2, 3,722 images, 118,572 annotations, 13 classes, official train/val/test split of 2,234 / 744 / 744 images. About 89% of boxes are smaller than 32x32 pixels.
- Detector (1280 px, test split): YOLO11m mAP50 0.252, mAP50-95 0.157; YOLO11s mAP50 0.237, mAP50-95 0.144. One training run each.
- Java inference on the Mac CPU: YOLO11m about 1.8 s per image, YOLO11s about 0.55 to 0.7 s per image.
- Learned tile scoring vs the class-weight heuristic (test, top 5 tiles per image, 610 images): ground-truth priority captured 0.785 vs 0.717, difference 0.068 with 95% interval [0.056, 0.082]. Random selection captures 0.194.
- Optimised route vs serpentine sweep over the same 5 tiles: shorter on 88.2% of images, longer on 0.2% (learned scoring).
- Cross-tile context (transformer) gave no measurable gain over a per-tile MLP (Spearman 0.661 vs 0.660).
- VAE does not beat PCA at 64 latent dimensions (reconstruction MSE 0.01193 vs 0.01132); the DCGAN produces blurry textures, no FID computed.

## Limitations

- Absolute detection accuracy is low: most objects are tiny, and about half of all ground-truth boxes get no detection at confidence 0.25 (docs/DETECTOR_ANALYSIS.md).
- The tile-priority target is built from ground-truth boxes and assumed class weights, so the planner evaluation measures agreement with that proxy, not real cleanup effort.
- Route lengths are in tile units from a fixed entry tile.
- Performance on photographs from outside the BePLi beaches has not been measured.
- No segmentation model was trained.

## Licences and data

BePLi v2 is CC BY-NC-SA 4.0 (non-commercial, share-alike). Ultralytics YOLO code and weights are AGPL-3.0. ONNX Runtime is MIT. This is a non-commercial academic project; check the licences before any redistribution.
