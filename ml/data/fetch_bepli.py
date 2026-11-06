"""Download and unpack the BePLi v2 archive (Colab only). Records size + SHA-256; never runs on your Mac by default.

The archive is about 8 GB (SEANOE record DOI 10.17882/96963, licence CC BY-NC-SA 4.0 as stated on the record).
Get the direct link from https://www.seanoe.org/data/00858/96963/ (Download button -> copy link) and pass it as --url:
this script deliberately has no hard-coded file URL because that URL was not verified.

Usage (in Colab):
  python -m ml.data.fetch_bepli --url "<copied link>" --dest /content/bepli_v2 --min-free-gb 25
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def free_gb(path: Path) -> float:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract a tar(.gz) refusing absolute paths / path traversal / links outside dest. Returns member count."""
    dest = Path(dest).resolve()
    n = 0
    with tarfile.open(archive) as tf:
        for m in tf:
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest)):
                raise ValueError(f"unsafe path in archive: {m.name}")
            if m.issym() or m.islnk():
                raise ValueError(f"links are not allowed in the archive: {m.name}")
            tf.extract(m, dest, filter="data")
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--min-free-gb", type=float, default=25.0, help="need ~8 GB archive + ~8+ GB extracted")
    ap.add_argument("--allow-mac", action="store_true", help="override the guard that blocks running on macOS")
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args(argv)

    if platform.system() == "Darwin" and not a.allow_mac:
        print("REFUSING: this downloads ~8 GB and extracts ~8+ GB; run it on Colab (or pass --allow-mac knowingly).")
        return 2
    dest = Path(a.dest)
    avail = free_gb(dest)
    if avail < a.min_free_gb:
        print(f"REFUSING: only {avail:.1f} GB free at {dest}, need {a.min_free_gb} GB.")
        return 2
    archive = dest / "bepli_v2.tar.gz"
    if not archive.exists():
        subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5", "-o", str(archive) + ".part", a.url], check=True)
        Path(str(archive) + ".part").rename(archive)
    digest = sha256_file(archive)
    record = {"url": a.url, "bytes": archive.stat().st_size, "sha256": digest,
              "downloaded": dt.datetime.now().isoformat(timespec="seconds"),
              "licence_per_record": "CC BY-NC-SA 4.0 (SEANOE 10.17882/96963); verify on the record before redistribution"}
    (dest / "DOWNLOAD_RECORD.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if not a.no_extract:
        n = safe_extract(archive, dest / "raw")
        print(f"extracted {n} members to {dest / 'raw'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
