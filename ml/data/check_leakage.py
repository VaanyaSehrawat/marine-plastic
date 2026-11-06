"""Verify a split file produced by make_splits.py. Exit code 0 = no leakage found, 1 = leakage, 2 = bad input.

Checks: every image in exactly one split; no group spans two splits; no cross-split pair closer than the dHash
threshold (recomputed from the images, not trusted from the file).

Usage: python -m ml.data.check_leakage --root DIR --splits data/splits/x.json [--threshold 6] [--cache data/interim/hashes.json]
       (--ann files are only needed to locate images; pass the same --ann values used for make_splits)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ml.data.coco_utils import build_image_index, load_json, resolve_image
from ml.data.near_duplicates import audit, compute_hashes


def check(splits_json: dict, split_hashes: dict[str, dict[str, int]], threshold: int) -> dict:
    problems = []
    splits = splits_json["splits"]
    where: dict[str, list[str]] = {}
    for s, names in splits.items():
        for n in names:
            where.setdefault(n, []).append(s)
    multi = [n for n, ss in where.items() if len(ss) > 1]
    if multi:
        problems.append(f"{len(multi)} images appear in more than one split, e.g. {multi[:3]}")
    group_splits: dict[int, set[str]] = {}
    for n, g in splits_json.get("group_of", {}).items():
        if n in where:
            group_splits.setdefault(g, set()).update(where[n])
    spanning = [g for g, ss in group_splits.items() if len(ss) > 1]
    if spanning:
        problems.append(f"{len(spanning)} groups span more than one split")
    rep = audit(split_hashes, threshold)
    for pair, v in rep["pairs"].items():
        if v["images_with_near_duplicate_in_other_split"]:
            problems.append(f"{pair}: {v['images_with_near_duplicate_in_other_split']} near-duplicate images across splits")
    return {"ok": not problems, "problems": problems, "audit": rep}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ann", action="append", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--cache", default="data/interim/hashes.json")
    a = ap.parse_args(argv)
    sj = json.loads(Path(a.splits).read_text())
    root = Path(a.root)
    index = build_image_index(root)
    path_of = {}
    for ann in a.ann:
        for im in load_json(Path(ann))["images"]:
            p = resolve_image(root, Path(ann), im["file_name"], index)
            if p is not None:
                path_of[im["file_name"]] = p
    hashes = compute_hashes(list(path_of.values()), Path(a.cache))
    split_hashes = {s: {n: hashes[str(path_of[n])] for n in names if n in path_of} for s, names in sj["splits"].items()}
    res = check(sj, split_hashes, a.threshold)
    print("LEAKAGE CHECK:", "PASS" if res["ok"] else "FAIL")
    for p in res["problems"]:
        print(" -", p)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
