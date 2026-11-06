# Milestone 3 - Dataset validation (Colab). Run only after M2 passes on the Mac.

Objective: decide, from the ACTUAL files, whether BePLi v2 is fit to train on; freeze leakage-safe splits and tile-label rules.
Tools exist (generated; tested on synthetic data only). Nothing has been run on real data.

## Files
`ml/data/{fetch_bepli,inspect_coco,near_duplicates,make_splits,check_leakage,tile_labels,prepare_subset,coco_utils}.py`,
`notebooks/01_bepli_colab_inspection.ipynb`, `configs/data.yaml`, `docs/DATASET_CARD.md`.

## Steps (Colab free, CPU runtime is enough)
1. Copy the project folder to Drive (`MyDrive/marine-plastic`). Open the notebook, run cells 1-2 (mount, install `pycocotools`).
2. Get the direct download link from https://www.seanoe.org/data/00858/96963/ (Copy link address on the Download button; the file URL is NOT hard-coded because it was not verified). Run the fetch cell: it needs >= 25 GB free on Colab local disk, records bytes + SHA-256, extracts safely to `/content/bepli_v2/raw`. Never to Drive, never to the Mac.
3. `inspect_coco`: real image/annotation/class counts compared with the claims in `configs/data.yaml` (MATCH/MISMATCH), missing files, decode failures, bbox/segmentation problems, class imbalance, quality heuristics.
4. **Manual review (mandatory)**: look at >= 30 random images with boxes (cell 4). Write answers in DATASET_CARD: are all plastic items annotated; are tiny fragments missed; do consecutive file numbers share a site; do different images show the same place.
5. Leakage: if the archive ships its own split files, audit them (`near_duplicates`). Then build our split (`make_splits`) and run `check_leakage` (must print PASS). Decide `--block-size` from step 4 (default 0 = off).
6. Tile-label statistics on a sample (cell 7): inspect a few positive / negative / uncertain tiles visually before accepting the rules.
7. Only then create the compact JPEG copy (cell 8) on Drive (Drive free quota is 15 GB total; expect ~1-2 GB, UNVERIFIED).
8. Bring `reports/*`, `data/splits/*.json` back to the Mac repo and update `HANDOFF.md` + `DATASET_CARD.md`: replace every UNVERIFIED mark you verified; keep marks you did not.

## Acceptance criteria (all must hold to "freeze the dataset")
- [ ] `reports/dataset_inspection.json` exists; images/classes/annotations measured; mismatches with claims explained in DATASET_CARD.
- [ ] 0 missing images, 0 decode failures in the sample (or documented and excluded).
- [ ] Visual review done and written down.
- [ ] `check_leakage` PASS on the split we use; audit of the official split (if any) recorded either way.
- [ ] Every class has >= 1 instance in train, val AND test (else merge/drop classes and record why).
- [ ] Tile-label counts (pos/neg/unc) recorded; uncertain tiles excluded from metrics and reported.
- [ ] `data/splits/bepli_v2_grouped.json` committed; test split declared untouched until M7.
Fallback if BePLi fails (license/files/quality): UAVVaste for detection only (single class, aerial; small) and say so plainly.

## Decision rules
- Official split shows > 0 near-duplicate cross-split images at dHash <= 6, or the nearest-neighbour histogram has a mass at <= 10: do not use it; use ours.
- If filename blocks are confirmed to share a site visit: set `block_size` to the typical visit length (largest plausible), re-run both split tools.
- Never merge UAVVaste/BeachLitter/SeaClear into training.

## After M3 (what Claude builds next)
M4 training pipeline (Python, Colab): dataset loader from compact COCO + frozen split, torchvision Faster R-CNN (ResNet50-FPN) training with Drive checkpoints, pycocotools evaluation on val (mAP@0.5, mAP@[.5:.95], per class), RetinaNet + tile classifiers (ResNet18, GoogLeNet) on the same split, ONNX export of the best detector with fixed-size tiles, and the real-model Python-vs-Java parity test. Written against the real measured class list from M3.
