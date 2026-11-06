"""Tile labelling rules (Milestone 3). Unannotated pixels are NOT automatically negative.

Given the union mask of all annotated plastic in an image and a tile grid:
  POSITIVE  (1): plastic covers >= pos_frac of the tile area.
  NEGATIVE  (0): NO plastic pixel inside the tile and none within `margin` pixels of it.
  UNCERTAIN (-1): everything else (a sliver of plastic, or plastic right at the tile boundary).
Uncertain tiles are excluded from training/metrics and reported separately. Negatives are only as reliable as the
dataset's annotation completeness, which is itself checked by manual review (docs/M3_DATASET_VALIDATION.md).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

POS, NEG, UNC = 1, 0, -1

try:
    from pycocotools import mask as mask_utils
except Exception:  # noqa: BLE001
    mask_utils = None


def tile_boxes(height: int, width: int, tile: int = 224, stride: int | None = None) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) boxes covering the image; edge tiles are shifted inwards so every tile is full size."""
    stride = stride or tile
    tile_h, tile_w = min(tile, height), min(tile, width)

    def starts(size, t):
        s = list(range(0, max(size - t, 0) + 1, stride))
        if s[-1] + t < size:
            s.append(size - t)
        return s

    return [(x, y, x + tile_w, y + tile_h) for y in starts(height, tile_h) for x in starts(width, tile_w)]


def union_mask(anns: list[dict], height: int, width: int) -> np.ndarray:
    """Binary mask of all annotated objects (polygons, RLE, or bbox fallback)."""
    mask = np.zeros((height, width), dtype=bool)
    for a in anns:
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg:
            img = Image.new("1", (width, height), 0)
            d = ImageDraw.Draw(img)
            for poly in seg:
                if len(poly) >= 6:
                    d.polygon([float(v) for v in poly], fill=1)
            mask |= np.asarray(img, dtype=bool)
        elif isinstance(seg, dict) and mask_utils is not None:
            rle = seg
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, height, width)
            mask |= mask_utils.decode(rle).astype(bool)
        elif a.get("bbox"):
            x, y, w, h = a["bbox"]
            mask[max(int(y), 0):min(int(y + h + 0.999), height), max(int(x), 0):min(int(x + w + 0.999), width)] = True
    return mask


def label_tiles(mask: np.ndarray, boxes, pos_frac: float = 0.02, margin: int = 16):
    """Returns (labels, fractions): labels in {1, 0, -1}; fractions = plastic pixel fraction per tile."""
    H, W = mask.shape
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def box_sum(x0, y0, x1, y1):
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])

    labels, fracs = [], []
    for (x0, y0, x1, y1) in boxes:
        area = max((x1 - x0) * (y1 - y0), 1)
        inside = box_sum(x0, y0, x1, y1)
        frac = inside / area
        fracs.append(frac)
        if frac >= pos_frac:
            labels.append(POS)
        elif box_sum(x0 - margin, y0 - margin, x1 + margin, y1 + margin) == 0:
            labels.append(NEG)
        else:
            labels.append(UNC)
    return np.array(labels, dtype=np.int8), np.array(fracs, dtype=np.float32)
