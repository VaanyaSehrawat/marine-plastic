from helpers import make_dataset

from ml.data import inspect_coco as ic


def test_inspector_counts_and_flags_defects(tmp_path):
    ds = make_dataset(tmp_path)
    rep = ic.inspect_dataset(ds["root"], list(ds["ann"].values()), sample_decode=50,
                             expect={"images": 21, "annotations": 40, "classes": 3})
    # 20 real images + 1 phantom entry; 40 normal annotations + 2 planted defects; unknown category 99 is not counted as a class
    assert rep["totals_across_files"] == {"images": 21, "annotations": 42, "classes_union": 3}
    assert rep["claims_check"]["images"]["status"] == "MATCH" and rep["claims_check"]["annotations"]["status"] == "MISMATCH"
    val = next(f for f in rep["files"] if f["annotation_file"] == "val.json")
    assert val["images_missing_on_disk"] == 1
    assert "bbox_out_of_bounds" in val["problems"]
    assert "annotation_unknown_category" in val["problems"]
    train = next(f for f in rep["files"] if f["annotation_file"] == "train.json")
    assert train["problems"] == {}
    assert train["sampled_decode_failures"] == 0 and train["images_without_annotations"] == 0


def test_claim_mismatch_is_reported(tmp_path):
    ds = make_dataset(tmp_path, with_defects=False)
    rep = ic.inspect_dataset(ds["root"], list(ds["ann"].values()), 10, {"images": 20, "annotations": 999})
    assert rep["claims_check"]["images"]["status"] == "MATCH"
    assert rep["claims_check"]["annotations"]["status"] == "MISMATCH"


def test_autodiscovery_and_markdown(tmp_path):
    ds = make_dataset(tmp_path, with_defects=False)
    rep = ic.inspect_dataset(ds["root"], None, 10, None)
    assert rep["n_annotation_files"] == 3
    md = ic.to_markdown(rep)
    assert "Dataset inspection report" in md and "unannotated" in md.lower()


def test_cli_writes_reports(tmp_path):
    ds = make_dataset(tmp_path / "d", with_defects=False)
    out = tmp_path / "rep" / "x"
    assert ic.main(["--root", str(ds["root"]), "--out", str(out), "--sample-decode", "5", "--expect-images", "20"]) == 0
    assert out.with_suffix(".json").exists() and out.with_suffix(".md").exists()
