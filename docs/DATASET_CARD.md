# Dataset card: BePLi Dataset v2 (updated 17 Nov 2026 with measured values)

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
