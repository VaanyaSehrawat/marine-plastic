# Milestone checklist (updated 17 Nov 2026)

Evidence: M = run on the owner's MacBook, K = run on Kaggle (T4 GPU), D = real BePLi data. Deadline 22 Nov 2026 (aim to submit 21 Nov). The original plan (5 Nov) used different detectors and extra items; where the plan changed, the row says what was actually done.

| ID | Milestone | What was done | Evidence |
|---|---|---|---|
| M1 | Environment | Project-local JDK 21.0.13 and Python 3.12.16, ONNX Runtime 1.20.0, 28 Python tests passed (1 skipped); re-checked by `scripts/verify_all.sh` | M |
| M2 | ONNX integration proof | Java ONNX Runtime smoke test passed on aarch64 (max difference about 1.2e-07) | M |
| M3 | Dataset validation | BePLi v2 downloaded and audited: counts, class totals, annotation checks, leakage audit, visual review (docs/DATASET_CARD.md). Official split kept; no grouped re-split; no tile labels | D |
| M4 | Detector trained | Changed from Faster R-CNN to YOLO11s and YOLO11m at 1280 px (DECISIONS D-08). Test mAP50-95 0.144 and 0.157; one run each | K, D |
| M4b | Second detector and tile classifiers | NOT DONE (RetinaNet, ResNet18, GoogLeNet were not trained) | - |
| M4c | ONNX export and real-model parity | Done on 3 fixture images instead of 20 tiles: boxes differ by at most 0.0212 px, scores by 2e-6, identical detection counts; Java reproduces them | M, D |
| M5 | Java application | Dashboard: image (sample or upload) to detections, tile priority, route order. Runs offline. No exported report file | M |
| M6 | Java fundamentals | Tensor, SVD, MLP with backpropagation, SGD, Momentum, Adam, Hebbian (Oja), competitive learning: 100 checks. Delta rule and RBM were NOT implemented | M |
| M7 | Final evaluation | Detector comparison, detector error analysis, planner evaluation, latency on 3 images (docs/RESULTS.md, DETECTOR_ANALYSIS.md, PLANNER_EVAL.md). NOT done: robustness tests (blur, brightness, JPEG), out-of-domain test, memory measurement. Validation split chose the detector; the test split was scored for both models | M, K, D |
| M8 | Documentation and demo | README, report draft, demo script, jury Q&A, deck, verification script, submission package. Still open: report [student] sections, recorded video, model card, clean-clone run | M |
| Stretch | Mask R-CNN, tile-context transformer, VAE | Transformer, VAE and DCGAN done; Mask R-CNN NOT done | M, K |
