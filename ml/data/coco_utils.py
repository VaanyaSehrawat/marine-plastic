"""Small helpers shared by the dataset tools: finding COCO files, resolving image paths, summary statistics."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def looks_like_coco(d) -> bool:
    return isinstance(d, dict) and all(k in d for k in ("images", "annotations", "categories"))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_coco_files(root: Path, max_mb: float = 2000.0) -> list[Path]:
    """Every *.json under root that parses as COCO (images/annotations/categories)."""
    found = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            if p.stat().st_size > max_mb * 1e6:
                continue
            if looks_like_coco(load_json(p)):
                found.append(p)
        except (OSError, ValueError):
            continue
    return found


def build_image_index(root: Path) -> dict[str, list[Path]]:
    """basename -> list of image files under root (used to resolve COCO file_name entries)."""
    index: dict[str, list[Path]] = {}
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            index.setdefault(p.name, []).append(p)
    return index


def resolve_image(root: Path, ann_path: Path, file_name: str, index: dict[str, list[Path]]) -> Path | None:
    """Find the image file for a COCO `file_name`. Tries common layouts, then a unique-basename match."""
    root = Path(root)
    candidates = [
        root / file_name,
        root / "images" / file_name,
        ann_path.parent / file_name,
        ann_path.parent / "images" / file_name,
        ann_path.parent.parent / "images" / file_name,
        ann_path.parent.parent / file_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    hits = index.get(Path(file_name).name, [])
    return hits[0] if len(hits) == 1 else None


def summarize(values) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0}

    def pct(q):
        return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]

    return {
        "n": len(vals), "min": vals[0], "p5": pct(0.05), "median": statistics.median(vals),
        "mean": round(statistics.fmean(vals), 4), "p95": pct(0.95), "max": vals[-1],
    }
