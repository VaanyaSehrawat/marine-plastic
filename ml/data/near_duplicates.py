"""Near-duplicate detection (perceptual dHash) and split-leakage audit (Milestone 3).

dHash: shrink to 9x8 grayscale, compare horizontal neighbours -> 64 bits. Hamming distance <= ~6 means almost the
same picture (re-encodes, tiny crops/noise). It does NOT detect "same place, different camera direction": that
needs the grouping rules in make_splits.py plus the manual review described in docs/M3_DATASET_VALIDATION.md.

Usage (audit the dataset's own splits; each --ann file is treated as one split named after the file):
  python -m ml.data.near_duplicates --root DIR --ann train.json --ann val.json --ann test.json \
      --threshold 6 --out reports/leakage_audit.json --cache data/interim/hashes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ml.data.coco_utils import build_image_index, load_json, resolve_image

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
BUCKETS = [(0, 0), (1, 4), (5, 8), (9, 12), (13, 16), (17, 64)]


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hash_file(path: Path) -> int:
    with Image.open(path) as im:
        return dhash(im)


def popcount64(x: np.ndarray) -> np.ndarray:
    return _POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def nearest(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For each hash in a, the minimum Hamming distance to any hash in b, and the index in b."""
    dist = np.empty(len(a), dtype=np.int32)
    idx = np.empty(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk):
        x = a[s:s + chunk, None] ^ b[None, :]
        d = popcount64(x)
        idx[s:s + chunk] = d.argmin(axis=1)
        dist[s:s + chunk] = d.min(axis=1)
    return dist, idx


def cluster(hashes: np.ndarray, threshold: int, chunk: int = 512) -> np.ndarray:
    """Union-find over all pairs with Hamming distance <= threshold. Returns a group label per hash."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, n, chunk):
        d = popcount64(hashes[s:s + chunk, None] ^ hashes[None, :])
        rows, cols = np.nonzero(d <= threshold)
        for r, c in zip(rows, cols):
            r += s
            if r != c:
                ra, rc = find(int(r)), find(int(c))
                if ra != rc:
                    parent[ra] = rc
    return np.array([find(i) for i in range(n)])


def compute_hashes(paths: list[Path], cache_file: Path | None = None) -> dict[str, int]:
    cache: dict[str, int] = {}
    if cache_file and Path(cache_file).exists():
        cache = {k: int(v) for k, v in json.loads(Path(cache_file).read_text()).items()}
    for p in paths:
        key = str(p)
        if key not in cache:
            cache[key] = hash_file(p)
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(json.dumps({k: str(v) for k, v in cache.items()}))
    return cache


def audit(split_hashes: dict[str, dict[str, int]], threshold: int = 6) -> dict:
    """split_hashes: split -> {image_key: hash}. Reports nearest cross-split distances (leakage evidence)."""
    arrays = {s: (list(h.keys()), np.array(list(h.values()), dtype=np.uint64)) for s, h in split_hashes.items() if h}
    report = {"threshold": threshold, "pairs": {}, "within_split_duplicate_pairs": {}}
    splits = list(arrays)
    for s in splits:  # within-split near duplicates (informational)
        keys, arr = arrays[s]
        lab = cluster(arr, threshold) if len(arr) > 1 else np.zeros(len(arr), dtype=int)
        report["within_split_duplicate_pairs"][s] = int(len(arr) - len(set(lab.tolist())))
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            ka, aa = arrays[a]
            kb, ab = arrays[b]
            dist, idx = nearest(aa, ab)
            hist = {f"{lo}-{hi}": int(((dist >= lo) & (dist <= hi)).sum()) for lo, hi in BUCKETS}
            order = np.argsort(dist)[:10]
            report["pairs"][f"{a} -> {b}"] = {
                "n_query": int(len(aa)),
                "images_with_near_duplicate_in_other_split": int((dist <= threshold).sum()),
                "fraction": round(float((dist <= threshold).mean()), 4),
                "nearest_distance_histogram": hist,
                "closest_examples": [{"a": ka[int(o)], "b": kb[int(idx[o])], "distance": int(dist[o])} for o in order],
            }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ann", action="append", required=True, help="COCO json per split (repeat)")
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--out", default="reports/leakage_audit.json")
    ap.add_argument("--cache", default="data/interim/hashes.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    index = build_image_index(root)
    split_hashes: dict[str, dict[str, int]] = {}
    for ann in a.ann:
        ap_ = Path(ann)
        data = load_json(ap_)
        paths = {im["file_name"]: resolve_image(root, ap_, im["file_name"], index) for im in data["images"]}
        found = {k: v for k, v in paths.items() if v is not None}
        hashes = compute_hashes(list(found.values()), Path(a.cache))
        split_hashes[ap_.stem] = {k: hashes[str(v)] for k, v in found.items()}
        print(f"{ap_.stem}: hashed {len(found)}/{len(paths)} images")
    rep = audit(split_hashes, a.threshold)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    for k, v in rep["pairs"].items():
        print(f"{k}: {v['images_with_near_duplicate_in_other_split']}/{v['n_query']} images have a near-duplicate "
              f"(dHash <= {a.threshold}) in the other split; histogram {v['nearest_distance_histogram']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
