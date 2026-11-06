import json

import numpy as np
from helpers import make_dataset

from ml.data import check_leakage, make_splits, near_duplicates as nd


def _hashes(ds):
    from ml.data.coco_utils import load_json
    out = {}
    for split, p in ds["ann"].items():
        data = load_json(p)
        out[split] = {im["file_name"]: nd.hash_file(ds["root"] / "images" / im["file_name"])
                      for im in data["images"] if (ds["root"] / "images" / im["file_name"]).exists()}
    return out


def test_dhash_identical_vs_different(tmp_path):
    ds = make_dataset(tmp_path)
    h = _hashes(ds)
    a, dup, other = h["train"]["t0.png"], h["test"]["s0.png"], h["train"]["t1.png"]
    assert bin(a ^ dup).count("1") <= 2
    assert bin(a ^ other).count("1") > 6


def test_audit_detects_planted_cross_split_duplicate(tmp_path):
    rep = nd.audit(_hashes(make_dataset(tmp_path)), threshold=6)
    pair = rep["pairs"]["train -> test"]
    assert pair["images_with_near_duplicate_in_other_split"] >= 1
    assert pair["closest_examples"][0]["distance"] <= 2


def test_splitter_keeps_duplicates_together_and_passes_check(tmp_path):
    ds = make_dataset(tmp_path, with_defects=False)
    out = tmp_path / "splits.json"
    anns = sum((["--ann", str(p)] for p in ds["ann"].values()), [])
    cache = str(tmp_path / "hash.json")
    assert make_splits.main(["--root", str(ds["root"]), *anns, "--out", str(out), "--seed", "1", "--cache", cache]) == 0
    sj = json.loads(out.read_text())
    where = {n: s for s, names in sj["splits"].items() for n in names}
    assert len(where) == 20 and where["t0.png"] == where["s0.png"]          # near-duplicates share a split
    assert check_leakage.main(["--root", str(ds["root"]), *anns, "--splits", str(out), "--cache", cache]) == 0


def test_leaky_split_is_rejected(tmp_path):
    ds = make_dataset(tmp_path, with_defects=False)
    anns = sum((["--ann", str(p)] for p in ds["ann"].values()), [])
    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps({"splits": {"train": ["t0.png", "t1.png"], "val": [], "test": ["s0.png"]}, "group_of": {}}))
    assert check_leakage.main(["--root", str(ds["root"]), *anns, "--splits", str(leaky),
                               "--cache", str(tmp_path / "h.json")]) == 1


def test_assignment_is_deterministic_and_roughly_proportional():
    groups = {f"img{i}.png": i // 2 for i in range(100)}
    a = make_splits.assign(groups, (0.7, 0.15, 0.15), seed=3)
    b = make_splits.assign(groups, (0.7, 0.15, 0.15), seed=3)
    assert a == b
    counts = {s: sum(1 for v in a.values() if v == s) for s in make_splits.SPLITS}
    assert 60 <= counts["train"] <= 80 and counts["val"] >= 10 and counts["test"] >= 10


def test_block_grouping_merges_consecutive_names():
    names = [f"{i:04d}.png" for i in range(10)]
    hashes = np.arange(10, dtype=np.uint64) * np.uint64(0x0101010101010101) * np.uint64(37)
    g = make_splits.build_groups(names, hashes, dup_threshold=0, block_size=5)
    assert len({g[n] for n in names[:5]}) == 1 and len({g[n] for n in names[5:]}) == 1
