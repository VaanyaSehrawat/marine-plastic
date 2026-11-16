#!/usr/bin/env bash
# Writes README.md, docs/REPORT.md, docs/DEMO_SCRIPT.md, docs/JURY_QA.md. Existing files are never overwritten (a .new copy is written instead).
# Place in ~/Projects/marine-plastic/scripts/ and run: bash scripts/m8_docs.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p docs build logs
put() {
  local f="$1" tmp; tmp="$(mktemp)"; cat > "$tmp"
  if [ -s "$f" ]; then mv "$tmp" "$f.new"; echo "  $f exists: wrote $f.new instead (kept the original)"
  else mv "$tmp" "$f"; echo "  wrote $f ($(wc -l < "$f" | tr -d ' ') lines)"; fi
}

put README.md <<'MDEOF1'
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

Fresh machine: `bash setup.sh` installs the project-local JDK 21 and Python 3.12 (see docs/README.md). The model weights are not stored in git; their SHA-256 values are in `models/*/SHA256.txt`.

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
MDEOF1

put docs/REPORT.md <<'MDEOF2'
# Marine Plastic Detection and Coastal Cleanup Planner: project report (draft of 16 Nov 2026)

Items marked [student] are for you to write in your own words after you have understood the section. Every number below comes from a generated results file in this repository.

## 1. Problem and goal

Plastic litter on beaches is hard to survey by hand. The goal is a system that finds litter in beach photographs, decides which parts of the image deserve attention first, and proposes an inspection order. [student: one paragraph on why this matters]

## 2. Requirements and how they are met

| requirement | status | where |
|---|---|---|
| Java application | done: Java 21 detector inference, planner and local dashboard | `java/app` |
| Reusable neural-network fundamentals module (tensors, SVD, MLP, optimizers, Hebbian, competitive learning) | done in pure Java, 100 automated checks | `java/fundamentals` |
| Trained vision pipeline | detection done (YOLO11s and YOLO11m); segmentation and classification-only models were not trained | `notebooks/`, `docs/RESULTS.md` |
| Transformer route-context module | done: 71,233-parameter transformer, exported to ONNX and run in Java | `docs/RESULTS.md`, `docs/PLANNER_EVAL.md` |
| GAN / VAE modules | both done offline in Python; no Java inference | `docs/RESULTS.md` |
| Evaluation package | detector comparison, error analysis, planner evaluation, parity and verification scripts | `docs/`, `scripts/verify_all.sh` |
| Live and recorded demonstration | script prepared | `docs/DEMO_SCRIPT.md` |

The hybrid design (Python training, Java application) was approved by the faculty.

## 3. Data: BePLi v2

3,722 images, 118,572 annotations, 13 classes, official split 2,234 / 744 / 744 images (71,889 / 24,643 / 22,040 annotations). Class imbalance is 65.2 to 1 (fragment 62,715 boxes, other_string 962). The median box is 14x9 pixels; 89% are smaller than 32x32 pixels.

Audit results: no missing files, orphan annotations, degenerate or out-of-bounds boxes; a segmentation mask for every annotation (not used). No filename or exact-duplicate overlap between splits. Perceptual-hash near-duplicate candidates exist (16 to 33 per split pair) but visual review showed different frames of similar scenes, and no test image met the combined rule (hash distance at most 5 and correlation at least 0.90) against train and validation. Images come from a limited set of beaches, so scene-level similarity between splits cannot be excluded.

## 4. Methods

### 4.1 Detector
YOLO11s and YOLO11m (Ultralytics 8.4.212, COCO-pretrained, fine-tuned) at 1280 pixels input, trained on a free Kaggle T4 GPU with AdamW, cosine learning rate, maximum 400 detections. The YOLO11s run was interrupted once and resumed; early stopping ended it at epoch 55 (best epoch 35). Exported to ONNX (opset 17) and run in Java with ONNX Runtime 1.20.0. The Java code implements its own letterbox resize, confidence filtering, class-aware non-maximum suppression and coordinate mapping. [student: what is letterboxing and why 1280 pixels]

### 4.2 Route-context transformer
Input: for each of 48 tiles (6x8 grid), 15 features computed from the detections (log of summed confidence per class, detection count, maximum confidence). Two transformer blocks (width 64, 4 heads) with learned tile-position embeddings; output: one priority per tile. Trained on 600 validation-split images (144 held out for early stopping), target = class-weighted ground-truth priority; compared against a per-tile MLP without cross-tile attention and against the detector-only heuristic.

### 4.3 Cleanup planner
Tiles are scored (heuristic: class weight times confidence; learned: transformer). Tiles above a threshold are visited in an order found by nearest neighbour plus 2-opt, starting from a fixed entry tile; a row-by-row serpentine sweep over the same tiles is the baseline. Class weights (nets and fishing gear 3.0, buoy 2.5, bottles, bags, containers and rope 2.0, string and foam 1.5, others 1.0, fragment 0.5) are assumptions, not measurements.

### 4.4 Fundamentals module (Java)
Tensor, one-sided Jacobi SVD, MLP with manual backpropagation, SGD / Momentum / Adam, Oja's Hebbian rule, winner-take-all competitive learning. Checks: SVD reconstruction error 1.06e-13 and orthogonality error 9.64e-15; MLP gradient check on 54 parameters, worst difference 3.90e-11; XOR learned with Adam in 39 steps (steps to loss below 0.05: SGD 112, Momentum 67, Adam 31); Oja's weight matches the first singular vector (cosine 0.99988); 3 clusters recovered with purity 100%. SVD image demo on a 240x320 photo: rank 5, 20, 60 give relative error 0.1938, 0.1214, 0.0700 using 3.7%, 14.6%, 43.8% of the numbers. [student: explain the SVD and Oja's rule]

### 4.5 VAE and DCGAN (offline)
64x64 crops of annotated objects (12,000 train, 3,000 validation). Convolutional VAE with a 64-dimensional latent space compared with PCA at the same size; DCGAN with a pixel-space nearest-neighbour copy check.

## 5. Results

### 5.1 Detectors (official splits, confidence 0.001, NMS IoU 0.6)

| model | split | precision | recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLO11s | val | 0.313 | 0.344 | 0.263 | 0.162 |
| YOLO11s | test | 0.299 | 0.301 | 0.237 | 0.144 |
| YOLO11m | val | 0.334 | 0.362 | 0.291 | 0.182 |
| YOLO11m | test | 0.307 | 0.333 | 0.252 | 0.157 |

Java CPU inference per image (MacBook M4, three fixture images): YOLO11s 549 to 694 ms, YOLO11m 1835 to 1879 ms. ONNX output agrees with PyTorch to 0.0212 pixels (boxes) and 2e-6 (scores) on the fixture images, and the Java detector reproduces the reference detections exactly (counts 93, 12, 13 for YOLO11m).

### 5.2 Detector error analysis (YOLO11m, test, confidence 0.25, IoU 0.5; own matching code)
- 31.5% of ground-truth boxes are correctly found and classified, 17.7% are localised but get the wrong class or lose the match, and 50.8% have no detection. At confidence 0.10, 70.7% are localised by some detection.
- Class-aware recall: 30.5% for small boxes (89.6% of all boxes), 40.0% for medium, 38.7% for large (only 271 large boxes).
- Weakest classes by own AP50: other_string 0.020, other_bottle 0.033, plastic_bag 0.052, others 0.078. For other_bottle, 55.4% of boxes are localised but only 9.5% of those get the right class, so classification, not localisation, is the main problem there. For fragment the class is usually right (85.7%) but only 47.3% of fragments are localised.
- Best classes: pet_bottle AP50 0.407, box_shaped_case 0.356, other_fishing_gear 0.321. [student: look at results/detectors/plots/yolo11m_test_confusion_matrix_normalized.png and name the two largest confusions]

### 5.3 Planner evaluation (test, 610 images with at least 5 ground-truth tiles, top K tiles per image)

| K | random | heuristic | learned | learned minus heuristic (95% CI) |
|---:|---:|---:|---:|---:|
| 3 | 0.174 | 0.704 | 0.763 | 0.058 [0.043, 0.073] |
| 5 | 0.194 | 0.717 | 0.785 | 0.068 [0.056, 0.082] |
| 8 | 0.242 | 0.743 | 0.819 | 0.076 [0.065, 0.087] |

(share of the best possible ground-truth priority captured). Optimised route vs serpentine sweep over the same tiles at K=5: heuristic 10.73 vs 14.35 tile units, learned 10.05 vs 13.63; the optimised route is shorter on 87.7% (heuristic) and 88.2% (learned) of images and longer on 0.8% and 0.2%.

Transformer vs per-tile MLP vs detector-only score (Spearman / top-5 overlap): 0.661 / 0.591, 0.660 / 0.590, 0.531 / 0.524; random -0.007 / 0.095. The cross-tile attention gave no measurable gain over the MLP.

### 5.4 VAE and DCGAN
VAE reconstruction MSE 0.01193 vs PCA 0.01132; 5-nearest-neighbour class accuracy in latent space 0.181 (VAE) and 0.190 (PCA), against a majority-class baseline of 0.197, so neither latent space carries much class information. DCGAN (80 epochs): generated samples have median nearest-training-crop distance 33.0 against 35.4 for real held-out crops, with no sample closer than every real crop (no sign of memorisation by this crude check). The sample grid shows blurry textures with a few object-like blobs. No FID was computed.

## 6. Limitations and threats to validity

- Small objects dominate; recall is low and absolute mAP is low. Tiling or higher-resolution inference might help but was not tested.
- One training run per model: the YOLO11s / YOLO11m gap is not a significance test.
- Planner evaluation uses a proxy target (ground-truth boxes with assumed class weights). The learned model is trained and scored on that same kind of target; the heuristic never saw it. Part of the gain may come from the transformer using detections down to confidence 0.10 while the heuristic uses 0.25; this was not separated.
- Route lengths are tile units from a fixed entry tile.
- Out-of-domain photographs (other beaches, phone cameras) were not evaluated.
- Scene-level similarity between splits is possible.
- Torchvision baselines (Faster R-CNN, RetinaNet) from the original plan were not trained; no comparison with them is claimed.

## 7. Engineering and reproducibility

Java and Python agree numerically (sections 5.1 and the route fixture: features 1.2e-07, transformer 3.6e-07, end-to-end priority correlation 1.0000). `bash scripts/verify_all.sh` re-runs the Java tests, dashboard self-tests, hash checks and re-generates the planner and detector analyses, checking that they reproduce the committed files. Training notebooks are in `notebooks/`. Weights are identified by SHA-256 (`models/*/SHA256.txt`).

## 8. Conclusion and next steps

[student: three honest sentences on what works, what does not, and what you would do next, for example tiling for small objects, evaluating on new photographs, and measuring the effect of the confidence floor on the transformer.]

## 9. Declarations

Data: BePLi v2 (CC BY-NC-SA 4.0). Code libraries: Ultralytics (AGPL-3.0), PyTorch, ONNX Runtime (MIT). AI assistants (Claude, ChatGPT) helped write code and documentation; follow your university's rule on how to state this. [student]
MDEOF2

put docs/DEMO_SCRIPT.md <<'MDEOF3'
# Demo script (about 7 minutes)

## Before the demo (do the day before and again 30 minutes before)

1. Plug in the laptop, close other heavy applications.
2. `cd ~/Projects/marine-plastic && bash scripts/verify_all.sh` and wait for `VERIFY: ALL CHECKS PASSED`.
3. Keep these tabs ready: the dashboard (http://127.0.0.1:8080), `docs/RESULTS.md`, `docs/DETECTOR_ANALYSIS.md`, `results/detectors/plots/yolo11m_test_confusion_matrix_normalized.png`, `results/generative/gan_samples.png`.
4. Have the recorded demo video and a copy of the repository on a USB stick.

## Live demo

| time | action | what to say (in your own words) |
|---|---|---|
| 0:00 | Start: `bash scripts/run_dashboard.sh` | The models were trained in Python on a free Kaggle GPU, exported to ONNX, and run here in Java with ONNX Runtime. |
| 0:45 | Choose `003278.png`, Scoring = Heuristic, Analyze | The detector finds litter boxes; each tile gets a score from class weights times confidence. The yellow route starts at an assumed entry tile in the top-left corner. |
| 2:00 | Switch Scoring to Learned (transformer), Analyze | A small transformer reads all tiles together and predicts a priority for each one. Different tiles get picked. |
| 3:00 | Move the Confidence and Tile threshold sliders | Lower confidence finds more objects but adds false ones. |
| 3:45 | Show the 7x9 grid with Learned selected | The transformer was trained for the 6x8 grid, so the dashboard falls back to the heuristic and says so in the panel. |
| 4:30 | Open `docs/PLANNER_EVAL.md` | Over 610 test images, learned scoring captures 0.785 of the best possible priority in the top 5 tiles, heuristic 0.717, random 0.194. The optimised route is shorter than a sweep on 88% of images. Priority is built from ground-truth boxes and assumed class weights, so this is agreement with a proxy. |
| 5:30 | Open `docs/DETECTOR_ANALYSIS.md` | About 90% of objects are tiny. Half of all objects get no detection at 0.25. Some classes are found but mislabelled. |
| 6:15 | Run `bash scripts/run_fundamentals.sh` or show its output | The Java fundamentals module has its own tensors, SVD, MLP and optimizers, with 100 automated checks. |
| 6:45 | Close with the limitations slide | Low absolute accuracy, one training run, proxy target, no out-of-domain test. |

Optional: upload your own beach photo (PNG or JPEG) through the dashboard. Say beforehand that performance on photographs outside the training beaches has not been measured.

## If something fails

- Dashboard slow or crashes: run `MODEL_DIR=yolo11s_1280 bash scripts/run_dashboard.sh` (about 3 times faster).
- Port 8080 busy: `lsof -ti tcp:8080 | xargs kill`, then start again.
- Java will not start: play the recorded video and show `docs/VERIFICATION.md`.

## Recorded demo

1. Press Cmd+Shift+5 and choose Record Entire Screen with the microphone on.
2. Follow the same table above, speaking slowly; keep it under 8 minutes.
3. Save as `demo_recording.mp4`. Do not commit it to git (large); keep it with the submission files.
MDEOF3

put docs/JURY_QA.md <<'MDEOF4'
# Likely jury questions and answers grounded in the measured results

Learn the idea behind each answer; do not memorise the wording.

1. **What did you build yourself and what did you use from libraries?** Written for this project: the Java fundamentals module, the Java detector pre- and post-processing (letterbox, non-maximum suppression), the transformer scorer, the planner, the dashboard server, the Python dataset conversion, the route transformer, the VAE, the DCGAN and the evaluation scripts. Used from libraries: Ultralytics for YOLO training, PyTorch, ONNX Runtime, NumPy. AI assistants helped write code; be ready to say so truthfully and check the university's disclosure rule.
2. **Why is detection accuracy so low?** About 89% of objects are smaller than 32x32 pixels and the median is 14x9 pixels; the dataset has 65 to 1 class imbalance. At confidence 0.25, 50.8% of ground-truth boxes have no detection at all (docs/DETECTOR_ANALYSIS.md).
3. **Is there data leakage between train and test?** No filename or exact-duplicate overlap. Near-duplicate candidates were checked visually; no test image met the strict rule. Scenes from the same beaches may still look alike, which is a remaining risk.
4. **Why YOLO and not the torchvision detectors in the original plan?** It was a time and engineering choice recorded in DECISIONS.md (D-08). We did not train Faster R-CNN or RetinaNet, so we claim no comparison with them.
5. **Why YOLO11m as the dashboard detector?** Higher validation mAP50-95 (0.182 vs 0.162) and test (0.157 vs 0.144). The cost is speed: about 1.8 s per image on the CPU against 0.55 to 0.7 s. One run each, so the gap is not a significance test.
6. **Is the validation-to-test gap a sign of overfitting?** YOLO11m drops from 0.182 to 0.157 mAP50-95. Validation was used to choose the model and for early stopping, so some drop is expected; the class mix also differs between splits (for example only 242 plastic bags in test against 700 in validation).
7. **Does the transformer help?** It beats the class-weight heuristic at capturing priority (0.785 vs 0.717 at K=5, interval [0.056, 0.082] for the difference). Against a per-tile MLP without attention it gives no measurable gain (Spearman 0.661 vs 0.660), so cross-tile context itself was not shown to matter.
8. **Is that comparison fair?** Not entirely. The transformer is trained on a target of the same kind it is scored on, the heuristic is not, and the transformer uses detections down to confidence 0.10 while the heuristic uses 0.25. We did not separate these effects.
9. **How were the class weights chosen?** They are assumptions (nets and fishing gear highest, fragments lowest), not measured; they define the priority target, so results depend on them.
10. **Is the route real navigation?** No. It is an inspection order over an image tile grid, in tile units from a fixed entry tile.
11. **Why is the optimised route better than the sweep, and always?** It starts from nearest neighbour and improves with 2-opt, so it adapts to which tiles were selected. It is shorter on 88.2% of images at K=5, equal on 11.6%, longer on 0.2%; it is not guaranteed to win.
12. **How do you know the Java detector matches the Python one?** Fixture tests on three images: identical detection counts (93, 12, 13), top-10 detections matched, ONNX versus PyTorch differences of 0.0212 pixels and 2e-6; the transformer features and outputs match to 1.2e-07 and 3.6e-07 (docs/VERIFICATION.md).
13. **Why Java at all?** The course requires a Java, C or C++ application; the hybrid Python-training and Java-inference design was approved by the faculty.
14. **What is letterboxing?** Resizing an image to fit a square input while keeping the aspect ratio and padding the rest with grey; boxes are mapped back by undoing the scale and padding.
15. **What is non-maximum suppression?** Keeping the highest-confidence box and removing overlapping boxes of the same class (IoU above 0.7 here).
16. **What do mAP50 and mAP50-95 mean?** Mean average precision over classes at IoU 0.5, and averaged over IoU 0.5 to 0.95. Small objects make high IoU hard, so mAP50-95 is much lower than mAP50.
17. **What is SVD and where did you use it?** Factorisation A = U S V^T. Implemented in Java (one-sided Jacobi), verified to 1e-13, used for the image compression demo and to check Oja's rule against the first singular vector.
18. **What does Oja's rule do?** A Hebbian update with weight decay that converges to the first principal direction; matched the SVD vector with cosine 0.99988.
19. **Why do the VAE and GAN not look good, and why are they in the project?** They are required modules and were trained on 64x64 crops. The VAE is no better than PCA (MSE 0.01193 vs 0.01132) and its latent space does not separate classes; the GAN makes blurry textures. No FID was computed, so we make no quality claim beyond visual inspection and the copy check.
20. **How would you improve it?** Evaluate tiling or higher-resolution inference for small objects, add out-of-domain tests, separate the effect of the confidence floor on the transformer, and use real distances for routes. These are untested ideas.
21. **What happens with a photo from another beach or a phone?** Not measured. The upload works, but accuracy there is unknown.
22. **Licences?** BePLi is CC BY-NC-SA 4.0 and Ultralytics is AGPL-3.0; this is a non-commercial academic project. Check them before any redistribution.
MDEOF4

echo "== commit and push =="
git add README.md docs/REPORT.md docs/DEMO_SCRIPT.md docs/JURY_QA.md scripts/m8_docs.sh scripts/verify_all.sh 2>/dev/null
BAD=0
for f in $(git diff --cached --name-only); do
  case "$f" in *.pt|*.pth|*.onnx|*.zip|*.tar.gz|*.mp4) echo "  FORBIDDEN staged file: $f"; BAD=1;; esac
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 3000000 ]; then echo "  FILE OVER 3 MB staged: $f"; BAD=1; fi
done
if [ $BAD -ne 0 ]; then git reset -q; echo "FAILED: unstaged everything, nothing committed."; exit 1; fi
if git diff --cached --quiet; then
  echo "  nothing new to commit"
else
  git commit -q -m "docs(m8): README, report draft, demo script and jury Q&A from the measured results" || { echo "FAILED: commit failed"; exit 1; }
  echo "  committed: $(git log --oneline | head -1)"
fi
git push > build/_push_docs.txt 2>&1; prc=$?
tail -3 build/_push_docs.txt | sed 's/^/  /'
[ $prc -eq 0 ] || { echo "COMMITTED LOCALLY, PUSH FAILED (exit $prc)"; exit 1; }
echo "  $(git status -sb | head -1)"
echo; echo "M8 DOCS: COMMITTED AND PUSHED"
