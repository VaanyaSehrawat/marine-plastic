"""Synthetic COCO dataset used by the dataset-tool tests (no real data is needed or implied)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 64, 48


def _pattern(rng, scale=8):
    small = rng.randint(0, 255, size=(H // scale, W // scale, 3)).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((W, H), Image.Resampling.BICUBIC))


def make_dataset(root: Path, seed: int = 0, with_defects: bool = True) -> dict:
    """train: t0..t11, val: v0..v3 (+defects), test: s0..s3 where s0 is a near-duplicate of t0."""
    rng = np.random.RandomState(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    layout = {"train": [f"t{i}.png" for i in range(12)], "val": [f"v{i}.png" for i in range(4)],
              "test": [f"s{i}.png" for i in range(4)]}
    arrays = {}
    for names in layout.values():
        for n in names:
            arrays[n] = _pattern(rng)
    noisy = arrays["t0.png"].copy()
    noisy[0, :3] = np.clip(noisy[0, :3].astype(int) + 1, 0, 255)
    arrays["s0.png"] = noisy
    for n, arr in arrays.items():
        Image.fromarray(arr).save(root / "images" / n)

    cats = [{"id": 1, "name": "bottle"}, {"id": 2, "name": "bag"}, {"id": 3, "name": "fragment"}]
    ann_paths, ann_id = {}, 1
    for split, names in layout.items():
        images, anns = [], []
        for k, n in enumerate(names):
            images.append({"id": k + 1, "file_name": n, "width": W, "height": H})
            for j in range(2):
                x, y, w, h = 4 + 10 * j, 5 + 3 * j, 12, 9
                anns.append({"id": ann_id, "image_id": k + 1, "category_id": 1 + (k + j) % 3, "bbox": [x, y, w, h],
                             "area": w * h, "iscrowd": 0,
                             "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]})
                ann_id += 1
        if split == "val" and with_defects:
            images.append({"id": 99, "file_name": "does_not_exist.png", "width": W, "height": H})
            anns.append({"id": ann_id, "image_id": 1, "category_id": 1, "bbox": [60, 40, 30, 30], "area": 900,
                         "iscrowd": 0, "segmentation": []}); ann_id += 1
            anns.append({"id": ann_id, "image_id": 2, "category_id": 99, "bbox": [1, 1, 5, 5], "area": 25,
                         "iscrowd": 0, "segmentation": []}); ann_id += 1
        p = root / f"{split}.json"
        p.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
        ann_paths[split] = p
    return {"root": root, "ann": ann_paths, "layout": layout}
