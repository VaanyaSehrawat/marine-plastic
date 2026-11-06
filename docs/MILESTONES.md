# Milestone checklist (pass/fail criteria)
Evidence levels: G generated | S sandbox-tested (Linux x86_64) | M tested on the owner's Mac | D tested on real data.
Deadline 22 Nov 2026 (aim to submit 21 Nov). Cut line 16 Nov; code freeze 18 Nov.

| ID | Milestone | Pass criteria | Evidence now |
|---|---|---|---|
| M1 | Environment | `bash setup.sh --minimal` ends with `SETUP COMPLETE`; JDK 21 arm64 and Python 3.12 inside `.tools/`/`.venv/`; system Java/Python unchanged | S (Linux dev mode); **Mac: not run** |
| M2 | ONNX integration proof | `bash scripts/run_tests.sh` ends `all tests passed`; Java prints `os.arch=aarch64`, `max|diff|` <= 1e-5+1e-4*|x| for batch 2 and 1, `[java] PASS`, then `ONNX smoke test PASSED` | Python half S; Java half: real ORT 1.20.0 API+engine on Linux S (JNI bridge built from official source, **not** the official jar); **Mac: not run** |
| M3 | Dataset validation (Colab) | Notebook 01 run: inspection report with measured counts vs claims; manual visual review done; leakage audit + `check_leakage` = PASS; splits frozen in `data/splits/`; DATASET_CARD updated with measured numbers; tile-label counts reviewed | Tools G + S on synthetic data only; **no real data touched** |
| M4 | Detector trained | Faster R-CNN trained on frozen train split; mAP@0.5 and mAP@[.5:.95] on **val** logged; test split untouched; checkpoints on Drive | not started |
| M4b | Second detector + tile classifiers | RetinaNet, ResNet18, GoogLeNet evaluated on the same val split | not started |
| M4c | ONNX export of best detector + real-model parity | PyTorch vs Java on 20 fixed tiles: boxes IoU >= 0.95, score diff <= 1e-3, same count above 0.5 | not started |
| M5 | Java application | Upload image -> detections -> tile priority -> route order -> report, from the dashboard; fallback demo data works offline | not started |
| M6 | Java fundamentals (essential demos) | `bash scripts/run_tests.sh` includes Java tests: tensor/SVD/gradcheck, MLP+backprop, optimizers, Hebbian/Oja/competitive/delta, RBM | not started |
| M7 | Final evaluation | One run on the held-out **test** split, robustness (blur/brightness/JPEG), UAVVaste shift test, latency/memory, error analysis | not started |
| M8 | Documentation + demo | README clean-clone verification, dataset card, model card, report, video, slides, viva Q&A | not started |
| Stretch | Mask R-CNN, tile-context transformer, VAE latent plot | only after M5 works end-to-end | not started |
