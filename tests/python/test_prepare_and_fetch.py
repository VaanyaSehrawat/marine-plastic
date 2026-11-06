import io
import json
import tarfile

import pytest
from helpers import make_dataset
from PIL import Image

from ml.data import fetch_bepli, prepare_subset


def test_prepare_subset_preserves_annotations_and_sizes(tmp_path):
    ds = make_dataset(tmp_path / "src", with_defects=False)
    out = tmp_path / "out"
    man = prepare_subset.prepare(ds["root"], ds["ann"]["train"], out, quality=90)
    assert man["images_kept"] == 12 and man["n_skipped"] == 0
    new = json.loads((out / "train.json").read_text())
    assert len(new["annotations"]) == 24
    with Image.open(out / new["images"][0]["file_name"]) as im:
        assert im.size == (64, 48)


def test_free_space_and_sha(tmp_path):
    assert fetch_bepli.free_gb(tmp_path) > 0
    f = tmp_path / "a.bin"; f.write_bytes(b"abc")
    assert fetch_bepli.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_safe_extract_blocks_traversal(tmp_path):
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tf:
        info = tarfile.TarInfo("../evil.txt"); data = b"x"; info.size = 1
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        fetch_bepli.safe_extract(bad, tmp_path / "dest")
    good = tmp_path / "good.tar"
    with tarfile.open(good, "w") as tf:
        info = tarfile.TarInfo("ok/file.txt"); info.size = 1
        tf.addfile(info, io.BytesIO(b"y"))
    assert fetch_bepli.safe_extract(good, tmp_path / "dest2") == 1


def test_fetch_refuses_when_not_enough_space(tmp_path):
    rc = fetch_bepli.main(["--url", "http://example.invalid/x", "--dest", str(tmp_path), "--min-free-gb", "99999999", "--allow-mac"])
    assert rc == 2
