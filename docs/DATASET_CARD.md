# Dataset card (DRAFT - to be completed with MEASURED values after Milestone 3)

| Field | Value | Status |
|---|---|---|
| Name | BePLi Dataset v2 (Beach Plastic Litter Dataset v2) | verified (record) |
| Source | SEANOE, DOI 10.17882/96963 | verified |
| Licence | CC BY-NC-SA 4.0 (non-commercial, share-alike) | verified on record; re-read before redistributing |
| Images | 3,722 claimed | UNVERIFIED -> measure |
| Annotations | ~118,572 claimed | UNVERIFIED -> measure |
| Classes | 13 plastic categories | UNVERIFIED names/counts -> measure |
| Format | COCO json, RLE instance masks | partly verified; layout UNVERIFIED |
| Domain | Ground-level beach photos, Yamagata, Japan, 2011-2019 | verified (v1 paper); no drones |
| Compact copy | JPEG q95 re-encode of PNGs (lossy) | to record after M3 |
| Splits | grouped by near-duplicate clusters; seed 42; ratios 0.70/0.15/0.15 | to record after M3 |
| Duplicates / leakage | audit result | to measure |
| Tile labels | positive / negative / uncertain (rules in `ml/data/tile_labels.py`) | counts to measure |
| Known limitations | single country/coast; photo directions repeat per site; classes imbalanced (to be measured); annotation completeness of tiny fragments unknown | |
| Out-of-domain test | UAVVaste (aerial, one generic class); image licence UNVERIFIED | optional |
