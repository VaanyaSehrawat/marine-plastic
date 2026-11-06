"""Leakage-aware train/val/test splitting (Milestone 3).

Images are first merged into GROUPS; a group is never split across train/val/test. Groups come from
  1. near-duplicate clusters (dHash Hamming distance <= --dup-threshold), and
  2. optional contiguous filename blocks (--block-size N): consecutive numeric file names are assumed to come from
     the same site visit. This is an ASSUMPTION about BePLi's naming that must be checked (docs/M3_DATASET_VALIDATION.md).
Groups are shuffled with --seed and assigned greedily to the split furthest below its target size.

Usage:
  python -m ml.data.make_splits --root DIR --ann all.json [--ann more.json] --out data/splits/bepli_v2_grouped.json \
      --ratios 0.7,0.15,0.15 --seed 42 --dup-threshold 6 --block-size 0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.data.coco_utils import build_image_index, load_json, resolve_image
from ml.data.near_duplicates import cluster, compute_hashes

SPLITS = ("train", "val", "test")


def numeric_key(name: str) -> int:
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[-1]) if nums else -1


def build_groups(names: list[str], hashes: np.ndarray, dup_threshold: int, block_size: int) -> dict[str, int]:
    """Return image name -> group id (union of duplicate clusters and optional filename blocks)."""
    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    lab = cluster(hashes, dup_threshold) if n > 1 else np.zeros(n, dtype=int)
    first: dict[int, int] = {}
    for i, l in enumerate(lab.tolist()):
        if l in first:
            union(i, first[l])
        else:
            first[l] = i
    if block_size and block_size > 0:
        order = sorted(range(n), key=lambda i: (numeric_key(names[i]), names[i]))
        for pos, i in enumerate(order):
            union(i, order[(pos // block_size) * block_size])
    return {names[i]: find(i) for i in range(n)}


def assign(groups: dict[str, int], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    members: dict[int, list[str]] = defaultdict(list)
    for name, g in groups.items():
        members[g].append(name)
    total = len(groups)
    target = {s: r * total for s, r in zip(SPLITS, ratios)}
    count = {s: 0 for s in SPLITS}
    gids = list(members)
    random.Random(seed).shuffle(gids)
    gids.sort(key=lambda g: -len(members[g]))          # big groups first (stable w.r.t. the shuffle)
    out: dict[str, str] = {}
    for g in gids:
        split = max(SPLITS, key=lambda s: (target[s] - count[s]) / max(target[s], 1e-9))
        for name in members[g]:
            out[name] = split
        count[split] += len(members[g])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ann", action="append", required=True)
    ap.add_argument("--out", default="data/splits/grouped_split.json")
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup-threshold", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=0)
    ap.add_argument("--cache", default="data/interim/hashes.json")
    a = ap.parse_args(argv)
    ratios = tuple(float(x) for x in a.ratios.split(","))
    assert len(ratios) == 3 and abs(sum(ratios) - 1) < 1e-6, "ratios must be three numbers summing to 1"

    root = Path(a.root)
    index = build_image_index(root)
    names, paths, class_by_image = [], [], defaultdict(Counter)
    seen = set()
    for ann in a.ann:
        data = load_json(Path(ann))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        id2name = {}
        for im in data["images"]:
            p = resolve_image(root, Path(ann), im["file_name"], index)
            if p is None or im["file_name"] in seen:
                continue
            seen.add(im["file_name"])
            names.append(im["file_name"]); paths.append(p); id2name[im["id"]] = im["file_name"]
        for an in data["annotations"]:
            if an["image_id"] in id2name:
                class_by_image[id2name[an["image_id"]]][cats.get(an["category_id"], "?")] += 1
    hashes_map = compute_hashes(paths, Path(a.cache))
    hashes = np.array([hashes_map[str(p)] for p in paths], dtype=np.uint64)

    groups = build_groups(names, hashes, a.dup_threshold, a.block_size)
    split_of = assign(groups, ratios, a.seed)
    result = {
        "meta": {"created": dt.datetime.now().isoformat(timespec="seconds"), "seed": a.seed, "ratios": ratios,
                 "dup_threshold": a.dup_threshold, "block_size": a.block_size, "n_images": len(names),
                 "n_groups": len(set(groups.values())),
                 "largest_group": max(Counter(groups.values()).values()) if groups else 0,
                 "annotation_files": [str(x) for x in a.ann]},
        "splits": {s: sorted(n for n, sp in split_of.items() if sp == s) for s in SPLITS},
        "group_of": groups,
        "class_instances_per_split": {s: dict(sum((class_by_image[n] for n, sp in split_of.items() if sp == s), Counter()))
                                      for s in SPLITS},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {a.out}: " + ", ".join(f"{s}={len(result['splits'][s])}" for s in SPLITS)
          + f"  (groups={result['meta']['n_groups']}, largest={result['meta']['largest_group']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
