"""Re-encode a COCO dataset into a compact copy (PNG -> JPEG) for Drive/Colab training (Milestone 3).

Only the image encoding changes: annotation geometry is untouched because image width/height are preserved.
JPEG is lossy (quality default 95); this is recorded in the output manifest and must be stated in the dataset card.

Usage: python -m ml.data.prepare_subset --root /content/bepli_v2/raw --ann train.json --out /content/drive/MyDrive/bepli_compact \
           [--quality 95] [--max-images 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from ml.data.coco_utils import build_image_index, load_json, resolve_image


def prepare(root: Path, ann: Path, out: Path, quality: int = 95, max_images: int = 0) -> dict:
    index = build_image_index(root)
    data = load_json(ann)
    (out / "images").mkdir(parents=True, exist_ok=True)
    kept, skipped, bytes_in, bytes_out = [], [], 0, 0
    for im in data["images"]:
        if max_images and len(kept) >= max_images:
            break
        src = resolve_image(root, ann, im["file_name"], index)
        if src is None:
            skipped.append(im["file_name"])
            continue
        new_name = Path(im["file_name"]).with_suffix(".jpg").name
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            if im.get("width") and (im["width"], im["height"]) != rgb.size:
                skipped.append(im["file_name"] + " (size differs from metadata)")
                continue
            rgb.save(out / "images" / new_name, "JPEG", quality=quality)
        bytes_in += src.stat().st_size
        bytes_out += (out / "images" / new_name).stat().st_size
        kept.append({**im, "file_name": f"images/{new_name}", "original_file_name": im["file_name"]})
    ids = {k["id"] for k in kept}
    new = {**data, "images": kept, "annotations": [a for a in data["annotations"] if a["image_id"] in ids]}
    (out / f"{ann.stem}.json").write_text(json.dumps(new))
    manifest = {"source_annotation": str(ann), "images_kept": len(kept), "images_skipped": skipped[:50],
                "n_skipped": len(skipped), "jpeg_quality": quality, "bytes_in": bytes_in, "bytes_out": bytes_out,
                "note": "lossy JPEG re-encode; annotations unchanged"}
    (out / f"{ann.stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--max-images", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(prepare(Path(a.root), Path(a.ann), Path(a.out), a.quality, a.max_images), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
