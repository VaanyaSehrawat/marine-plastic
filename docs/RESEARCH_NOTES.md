# Research notes (so no one needs to repeat the research)
Compiled 2026-11-05. "Verified" = read from the official record/source. Everything else is marked UNVERIFIED.

## ONNX Runtime Java on Apple Silicon
- Java artifact `com.microsoft.onnxruntime:onnxruntime` on Maven Central. Listed 1.20.x versions: **1.20.0 only** (no 1.20.1). Owner's Mac confirmed the 404 for 1.20.1 and success for 1.20.0.
- The official docs page still lists macOS x64 only for the CPU artifact; it is stale. Source at tag v1.20.0: the packaging pipeline merges a macOS ARM64 Java artifact into the final jar and the packaging script renames `osx-arm64` -> `osx-aarch64`; the loader (`OnnxRuntime.java`) computes `osx-aarch64`. Owner's Mac then confirmed both dylibs inside the jar.
- Sandbox: Java API compiled from official v1.20.0 source + JNI bridge built from the official C sources against the real libonnxruntime 1.20.0 (from the Python wheel) ran the smoke test on Linux x64. Not the official jar; not a Mac test.

## Datasets (primary: BePLi v2)
| Dataset | Verified facts | Decision |
|---|---|---|
| BePLi v2 (SEANOE, DOI 10.17882/96963) | CC BY-NC-SA 4.0; 3,722 images; 13 plastic classes; COCO json with RLE instance masks; one ~8 GB tar.gz; beach photos, Yamagata (Japan), 2011-2019. From the v1 paper (same images family): photographed from 3-4 directions per site plus close-ups; median size 743x557; ~32 annotations/image (119,192 in v1); official v1 split 2,226/742/741; site location withheld. | Primary (provisional) |
| BeachLitter v2022 | CC BY-NC-SA at SEANOE (paper text is CC BY); 3,500 images, Yamagata 2011-2019, 8 semantic classes ("artificial litter" etc.) | Not used (not plastic-specific; probable overlap with BePLi) |
| UAVVaste | Repo Apache-2.0; ~772 images, ~3,718 annotations, one generic class; image licence and plastic specificity UNVERIFIED | Out-of-domain test only |
| SeaClear | CC BY 4.0; 8,610 images, 40 categories, underwater ROV, 1920x1080, ~1.7 GB | Not used |
UNVERIFIED: BePLi v2 annotation count (~118,572), per-class counts, archive layout, whether consecutive file numbers share a site, whether the official v2 split leaks.

## Why leakage is the main data risk
Same site photographed from several directions and across years => near-duplicate or overlapping scenes. Site IDs are hidden, so grouping uses dHash near-duplicate clusters (+ optional filename blocks) and an audit of the official split.
