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
