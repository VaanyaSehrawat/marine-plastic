"""Dataset inspection for COCO-format datasets (Milestone 3).

Verifies, from the ACTUAL files: images exist and decode, annotation/category consistency, per-class counts,
annotations per image, bounding-box sanity, segmentation types (and RLE decoding when pycocotools is present),
image size and simple quality heuristics, duplicate ids/file names, and overlap between annotation files.
Claimed statistics (from papers or summaries) can be passed with --expect-* and are reported MATCH/MISMATCH.
It never treats an unannotated region as a negative sample: it only reports what is in the files.

Usage:
  python -m ml.data.inspect_coco --root /content/bepli_v2 --out reports/dataset_inspection \
      [--sample-decode 300] [--expect-images 3722 --expect-classes 13 --expect-annotations 118572]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ml.data.coco_utils import build_image_index, find_coco_files, load_json, resolve_image, summarize

try:  # optional; only needed to decode RLE masks
    from pycocotools import mask as mask_utils
except Exception:  # noqa: BLE001
    mask_utils = None


def image_quality(arr_gray: np.ndarray) -> dict:
    """Cheap heuristics on a grayscale array: brightness, contrast, sharpness (variance of a Laplacian)."""
    g = arr_gray.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return {"brightness": float(g.mean()), "contrast": float(g.std()), "sharpness": float(lap.var())}


def inspect_coco_file(ann_path: Path, root: Path, index: dict, sample_decode: int = 200, seed: int = 0) -> dict:
    data = load_json(ann_path)
    images, anns, cats = data["images"], data["annotations"], data["categories"]
    try:
        name = str(ann_path.relative_to(root))
    except ValueError:
        name = str(ann_path)
    rep: dict = {"annotation_file": name, "n_images": len(images), "n_annotations": len(anns), "n_categories": len(cats)}
    problems: dict[str, list] = defaultdict(list)

    ids = [im["id"] for im in images]
    names = [im["file_name"] for im in images]
    rep["duplicate_image_ids"] = len(ids) - len(set(ids))
    rep["duplicate_file_names"] = len(names) - len(set(names))
    cat_names = {c["id"]: c.get("name", str(c["id"])) for c in cats}
    img_by_id = {im["id"]: im for im in images}

    per_class: Counter = Counter()
    per_image: Counter = Counter()
    seg_types: Counter = Counter()
    areas, crowd = [], 0
    for a in anns:
        iid, cid = a.get("image_id"), a.get("category_id")
        if iid not in img_by_id:
            problems["annotation_unknown_image"].append(a.get("id"))
            continue
        if cid not in cat_names:
            problems["annotation_unknown_category"].append(a.get("id"))
        per_class[cat_names.get(cid, f"?{cid}")] += 1
        per_image[iid] += 1
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            problems["bbox_missing"].append(a.get("id"))
        else:
            x, y, w, h = bbox
            im = img_by_id[iid]
            if w <= 0 or h <= 0:
                problems["bbox_nonpositive"].append(a.get("id"))
            W, H = im.get("width"), im.get("height")
            if W and H and (x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1):
                problems["bbox_out_of_bounds"].append(a.get("id"))
            areas.append(max(w, 0) * max(h, 0))
        seg = a.get("segmentation")
        seg_types["none" if not seg else ("polygon" if isinstance(seg, list) else "rle")] += 1
        crowd += int(a.get("iscrowd", 0) or 0)

    counts = [per_image.get(i, 0) for i in ids]
    rep["annotations_per_image"] = summarize(counts)
    rep["images_without_annotations"] = sum(1 for c in counts if c == 0)
    rep["per_class_instances"] = dict(sorted(per_class.items(), key=lambda kv: -kv[1]))
    rep["class_imbalance_ratio_max_over_min"] = (
        round(max(per_class.values()) / max(1, min(per_class.values())), 2) if per_class else None)
    rep["categories_with_zero_instances"] = [n for n in cat_names.values() if per_class.get(n, 0) == 0]
    rep["segmentation_types"] = dict(seg_types)
    rep["iscrowd_count"] = crowd
    rep["bbox_area_px"] = summarize(areas)
    rep["image_size_from_metadata"] = {
        "width": summarize([im.get("width", 0) for im in images if im.get("width")]),
        "height": summarize([im.get("height", 0) for im in images if im.get("height")]),
    }

    # files on disk (all images: existence); decode + quality on a random sample
    missing, resolved = [], {}
    for im in images:
        p = resolve_image(root, ann_path, im["file_name"], index)
        if p is None:
            missing.append(im["file_name"])
        else:
            resolved[im["id"]] = p
    rep["images_missing_on_disk"] = len(missing)
    if missing:
        problems["image_file_missing"] = missing[:50]

    rng = random.Random(seed)
    sample_ids = rng.sample(list(resolved), min(sample_decode, len(resolved)))
    decode_fail, size_mismatch, quality, file_kb = [], [], [], []
    for iid in sample_ids:
        p = resolved[iid]
        try:
            with Image.open(p) as im_:
                im_.load()
                w, h = im_.size
                gray = np.asarray(im_.convert("L"))
            meta = img_by_id[iid]
            if meta.get("width") and meta.get("height") and (meta["width"], meta["height"]) != (w, h):
                size_mismatch.append(meta["file_name"])
            quality.append(image_quality(gray) if min(gray.shape) > 3 else {"brightness": 0, "contrast": 0, "sharpness": 0})
            file_kb.append(p.stat().st_size / 1024)
        except Exception as exc:  # noqa: BLE001
            decode_fail.append(f"{p.name}: {exc}")
    rep["sampled_images_decoded"] = len(sample_ids) - len(decode_fail)
    rep["sampled_decode_failures"] = len(decode_fail)
    rep["sampled_size_mismatch_vs_metadata"] = len(size_mismatch)
    if decode_fail:
        problems["image_decode_failure"] = decode_fail[:20]
    if size_mismatch:
        problems["image_size_mismatch"] = size_mismatch[:20]
    rep["sample_file_size_kb"] = summarize(file_kb)
    for key in ("brightness", "contrast", "sharpness"):
        rep[f"sample_{key}"] = summarize([q[key] for q in quality])
    rep["sample_flagged_dark_or_bright"] = sum(1 for q in quality if q["brightness"] < 30 or q["brightness"] > 225)
    rep["sample_flagged_low_contrast"] = sum(1 for q in quality if q["contrast"] < 10)
    rep["sample_flagged_blurry_heuristic"] = sum(1 for q in quality if q["sharpness"] < 5)

    # RLE decoding check
    rle_anns = [a for a in anns if isinstance(a.get("segmentation"), dict)]
    if rle_anns:
        if mask_utils is None:
            rep["rle_decode_check"] = "skipped (pycocotools not installed)"
        else:
            bad = 0
            for a in rng.sample(rle_anns, min(300, len(rle_anns))):
                try:
                    seg = a["segmentation"]
                    if isinstance(seg.get("counts"), list):
                        im = img_by_id[a["image_id"]]
                        seg = mask_utils.frPyObjects(seg, im["height"], im["width"])
                    mask_utils.decode(seg)
                except Exception:  # noqa: BLE001
                    bad += 1
            rep["rle_decode_check"] = {"sampled": min(300, len(rle_anns)), "failed": bad}
            if bad:
                problems["rle_decode_failure"] = [bad]
    rep["problems"] = {k: {"count": len(v), "examples": v[:5]} for k, v in problems.items()}
    rep["_file_names"] = names
    return rep


def inspect_dataset(root: Path, ann_files: list[Path] | None = None, sample_decode: int = 200, expect: dict | None = None) -> dict:
    root = Path(root)
    ann_files = ann_files or find_coco_files(root)
    index = build_image_index(root)
    files = [inspect_coco_file(p, root, index, sample_decode) for p in ann_files]
    names_per_file = {f["annotation_file"]: set(f.pop("_file_names")) for f in files}
    overlaps = {}
    keys = list(names_per_file)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            common = names_per_file[keys[i]] & names_per_file[keys[j]]
            if common:
                overlaps[f"{keys[i]} <-> {keys[j]}"] = len(common)
    totals = {
        "images": sum(f["n_images"] for f in files),
        "annotations": sum(f["n_annotations"] for f in files),
        "classes_union": len({c for f in files for c in f["per_class_instances"] if not c.startswith("?")}),
    }
    rep = {"root": str(root), "n_annotation_files": len(files), "totals_across_files": totals,
           "file_name_overlap_between_annotation_files": overlaps, "files": files,
           "claims_check": {}}
    for key, got in (("images", totals["images"]), ("annotations", totals["annotations"]), ("classes", totals["classes_union"])):
        want = (expect or {}).get(key)
        if want is not None:
            rep["claims_check"][key] = {"claimed": want, "measured": got, "status": "MATCH" if want == got else "MISMATCH"}
    return rep


def to_markdown(rep: dict) -> str:
    L = ["# Dataset inspection report", "", f"Root: `{rep['root']}`  ", f"Annotation files: {rep['n_annotation_files']}", ""]
    t = rep["totals_across_files"]
    L += [f"Totals across files (measured): images={t['images']}, annotations={t['annotations']}, classes={t['classes_union']}", ""]
    if rep["claims_check"]:
        L += ["## Claimed vs measured", "", "| statistic | claimed | measured | status |", "|---|---|---|---|"]
        for k, v in rep["claims_check"].items():
            L.append(f"| {k} | {v['claimed']} | {v['measured']} | {v['status']} |")
        L.append("")
    if rep["file_name_overlap_between_annotation_files"]:
        L += ["## WARNING: same file_name in more than one annotation file", ""]
        L += [f"- {k}: {v}" for k, v in rep["file_name_overlap_between_annotation_files"].items()] + [""]
    for f in rep["files"]:
        L += [f"## {f['annotation_file']}", "",
              f"- images: {f['n_images']}, annotations: {f['n_annotations']}, categories: {f['n_categories']}",
              f"- images without annotations: {f['images_without_annotations']}",
              f"- annotations per image: {f['annotations_per_image']}",
              f"- images missing on disk: {f['images_missing_on_disk']}; sampled decode failures: {f['sampled_decode_failures']}",
              f"- segmentation types: {f['segmentation_types']}; iscrowd: {f['iscrowd_count']}",
              f"- class imbalance (max/min): {f['class_imbalance_ratio_max_over_min']}",
              f"- sample quality flags: dark/bright={f['sample_flagged_dark_or_bright']}, low-contrast={f['sample_flagged_low_contrast']}, blurry(heuristic)={f['sample_flagged_blurry_heuristic']}",
              "", "| class | instances |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in f["per_class_instances"].items()]
        L.append("")
        if f["problems"]:
            L += ["**Problems found:**", ""] + [f"- {k}: {v['count']} (e.g. {v['examples']})" for k, v in f["problems"].items()] + [""]
        else:
            L += ["No structural problems found.", ""]
    L += ["_Quality flags are heuristics, not labels. Nothing here treats unannotated regions as negatives._"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ann", action="append", help="COCO json (repeatable). Default: auto-discover under --root")
    ap.add_argument("--out", default="reports/dataset_inspection")
    ap.add_argument("--sample-decode", type=int, default=200)
    ap.add_argument("--expect-images", type=int)
    ap.add_argument("--expect-annotations", type=int)
    ap.add_argument("--expect-classes", type=int)
    a = ap.parse_args(argv)
    expect = {"images": a.expect_images, "annotations": a.expect_annotations, "classes": a.expect_classes}
    rep = inspect_dataset(Path(a.root), [Path(p) for p in a.ann] if a.ann else None, a.sample_decode, expect)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    out.with_suffix(".md").write_text(to_markdown(rep))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    print(f"totals: {rep['totals_across_files']}")
    for k, v in rep["claims_check"].items():
        print(f"claim {k}: claimed={v['claimed']} measured={v['measured']} -> {v['status']}")
    n_prob = sum(len(f["problems"]) for f in rep["files"])
    print(f"problem categories found: {n_prob}")
    return 0 if rep["n_annotation_files"] else 2


if __name__ == "__main__":
    sys.exit(main())
