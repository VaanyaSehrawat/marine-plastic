import pathlib

def read(p): return pathlib.Path(p).read_text()
def write(p, s): pathlib.Path(p).write_text(s)

def banner(path, marker, text):
    s = read(path)
    if marker in s:
        print(f"  {path}: banner already present"); return
    lines = s.split("\n")
    lines[1:1] = ["", text, ""]
    write(path, "\n".join(lines)); print(f"  {path}: banner added")

# ---------------- docs/DATASET_CARD.md (replaced) ----------------
CARD = """# Dataset card: BePLi Dataset v2 (updated 17 Nov 2026 with measured values)

| Field | Value | Status |
|---|---|---|
| Name | BePLi Dataset v2 (Beach Plastic Litter Dataset v2) | verified (record) |
| Source | SEANOE, DOI 10.17882/96963 | verified |
| Licence | CC BY-NC-SA 4.0 (non-commercial, share-alike) | verified on record; re-read before redistributing |
| Images | 3,722 (train 2,234 / val 744 / test 744) | measured |
| Annotations | 118,572 (train 71,889 / val 24,643 / test 22,040) | measured |
| Classes | 13: pet_bottle, other_bottle, plastic_bag, box_shaped_case, other_container, rope, other_string, fishing_net, buoy, other_fishing_gear, styrene_foam, others, fragment | measured |
| Format | COCO JSON in `annotation/` (one file per split), images in `images/train`, `images/val`, `images/test`; a segmentation entry exists for every annotation (masks not used in this project) | measured |
| Domain | Ground-level beach photos, Yamagata, Japan, 2011-2019 | from the v1 paper; not re-verified |
| Image sizes | 116 distinct sizes in train; most common 743x557 (667 images) | measured |
| Annotation checks | 0 missing files, 0 orphan annotations, 0 degenerate or out-of-bounds boxes, 0 crowd annotations | measured |
| Object size | Median box 14x9 px; boxes under 32x32 px: 89.2% (train), 89.3% (val), 89.6% (test) | measured |
| Objects per image | Median 20 / 22 / 16 (train / val / test); maximum 363 / 319 / 226 | measured |
| Class imbalance | 65.2 to 1 (fragment 62,715 vs other_string 962) | measured |
| Splits | Official BePLi v2 split used unchanged | decision; no grouped re-split was made |
| Leakage audit | No filename overlap and no exact (MD5) duplicates between splits. Perceptual-hash candidates at distance 5 or less: 17 val images with a train match, 33 test images with a train match, 16 test images with a val match (of 744 each). Visual review of the closest pairs showed different frames of similar scenes; one val/train pair (val 001350, train 001349) looks like consecutive frames. The combined rule (hash distance at most 5 and correlation at least 0.90) flagged 0 test images against train and validation. | measured |
| Tile labels | Not used. Planner priority is built from ground-truth boxes weighted by assumed class weights | decision |
| Compact copy | Not made; full images were used | decision |
| Out-of-domain test | Not done | not done |
| Known limitations | Single country and coast; similar scenes recur across splits, so scene-level leakage cannot be excluded; strong class imbalance; 89% of boxes are tiny; annotation completeness for tiny fragments is unknown | |

## Class counts (measured)

| class | train | val | test | total |
|---|---:|---:|---:|---:|
| fragment | 38,938 | 13,362 | 10,415 | 62,715 |
| others | 13,932 | 4,564 | 4,693 | 23,189 |
| styrene_foam | 5,924 | 2,024 | 2,610 | 10,558 |
| pet_bottle | 2,740 | 934 | 997 | 4,671 |
| buoy | 1,503 | 500 | 887 | 2,890 |
| plastic_bag | 1,848 | 700 | 242 | 2,790 |
| rope | 1,656 | 635 | 452 | 2,743 |
| other_bottle | 1,488 | 549 | 514 | 2,551 |
| fishing_net | 1,242 | 393 | 323 | 1,958 |
| other_container | 740 | 300 | 327 | 1,367 |
| box_shaped_case | 630 | 209 | 262 | 1,101 |
| other_fishing_gear | 631 | 261 | 185 | 1,077 |
| other_string | 617 | 212 | 133 | 962 |

Sources: notebooks/ (dataset review) and docs/RESULTS.md. The class mix differs between splits (for example plastic_bag: 700 val, 242 test), which affects per-class comparisons.
"""
write("docs/DATASET_CARD.md", CARD); print("  docs/DATASET_CARD.md: replaced")

# ---------------- docs/MILESTONES.md (replaced) ----------------
MS = """# Milestone checklist (updated 17 Nov 2026)

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
"""
write("docs/MILESTONES.md", MS); print("  docs/MILESTONES.md: replaced")

# ---------------- HANDOFF.md ----------------
h = read("HANDOFF.md")
BAN = "> **Historical sections:** the status table and \"Known issues\" below were written on 5 Nov 2026 before any real data was used. They are superseded by the section \"Status 17 Nov 2026 (final)\" at the end of this file."
if "Historical sections:" not in h:
    lines = h.split("\n"); lines[1:1] = ["", BAN, ""]; h = "\n".join(lines)
for old, new in (("## Status", "## Status (historical, 5 Nov 2026)"), ("## Known issues / unverified", "## Known issues / unverified (historical, 5 Nov 2026)"),
                 ("## Next tasks (in order)", "## Next tasks (historical, 5 Nov 2026)")):
    h = "\n".join(new if l == old else l for l in h.split("\n"))
if "Status 17 Nov 2026 (final)" not in h:
    h += """

## Status 17 Nov 2026 (final)
- Done and verified: environment, BePLi v2 audit, YOLO11s and YOLO11m (Kaggle), ONNX export with parity, Java detector, route-context transformer in Java, cleanup planner and dashboard, Java fundamentals, VAE and DCGAN, planner evaluation, detector error analysis. `bash scripts/verify_all.sh` passes 12 of 12 checks (docs/VERIFICATION.md).
- Not done: Faster R-CNN and RetinaNet baselines, segmentation, tile classifiers, robustness and out-of-domain tests, delta rule and RBM.
- Dataset facts are measured, not claimed: see docs/DATASET_CARD.md. Milestones: docs/MILESTONES.md. Decisions: DECISIONS.md (D-08, D-09 and the 17 Nov resolution table).
- Open for the student: write the [student] sections of docs/REPORT.md, rehearse and record the demo, state AI-assistant help per the university rule, submit by 22 Nov (aim 21 Nov).
"""
write("HANDOFF.md", h); print("  HANDOFF.md: banner, headings and final status")

# ---------------- DECISIONS.md (append only) ----------------
d = read("DECISIONS.md")
if "Resolution of open and provisional items" not in d:
    d += """

## Update 17 Nov 2026 - Resolution of open and provisional items
| ID | Outcome |
|---|---|
| D-06 | Superseded by D-08: YOLO11 was used; torchvision Faster R-CNN and RetinaNet were not trained |
| P-01 | Confirmed: BePLi v2 was the only dataset used; no unusable annotations found |
| P-02 | Not done: UAVVaste was not used |
| P-03 | Unchanged: BeachLitter and SeaClear were not used |
| P-04 | Not needed: the official split was kept; audit found no filename or exact-duplicate overlap and no test image flagged by the combined near-duplicate rule |
| P-05 | Not used: planner priority comes from ground-truth boxes with assumed class weights, not tile labels |
| P-06 | Unchanged: non-commercial academic use; re-read the licences before any redistribution |
| O-01 | Unresolved: whether consecutive file numbers share a site was not determined |
| O-02 | Resolved: archive https://www.seanoe.org/data/00858/96963/data/106157.tar.gz extracts to `BePLi_dataset_v2/plastic_coco/` with `annotation/` and `images/{train,val,test}` |
| O-03 | Moot: both a VAE and a DCGAN were built |
| O-04 | Out: segmentation was not trained |
| O-05 | Resolved: 118,572 annotations and 3,722 images, as measured (docs/DATASET_CARD.md) |
"""
    write("DECISIONS.md", d); print("  DECISIONS.md: resolution table appended")
else:
    print("  DECISIONS.md: resolution table already present")

# ---------------- banners ----------------
banner("docs/RESEARCH_NOTES.md", "Later verified:", "> **Later verified:** the BePLi v2 counts, per-class counts, archive layout and split audit marked UNVERIFIED below were measured afterwards; see docs/DATASET_CARD.md. The notes below are the 5 Nov 2026 snapshot.")
banner("docs/M3_DATASET_VALIDATION.md", "Historical procedure (5 Nov 2026)", "> **Historical procedure (5 Nov 2026):** what was actually done is in docs/DATASET_CARD.md and notebooks/INDEX.md. The compact JPEG copy and grouped re-split described below were not made; the official split and full images were used.")
banner("docs/SETUP.md", "Historical status line:", "> **Historical status line:** the status text below dates from 5 Nov 2026. Milestones 1 and 2 were later verified on the Mac; see docs/MILESTONES.md.")
print("UPDATE: DONE")
