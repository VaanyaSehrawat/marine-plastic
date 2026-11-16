import re, zipfile, pathlib

OUT = pathlib.Path("results/detectors/plots")
OUT.mkdir(parents=True, exist_ok=True)
SPEC = [("backups/marine_run2.zip", {"eval_m_test": "yolo11m_test", "eval_s_test": "yolo11s_test", "yolo11m_1280": "yolo11m_training"}),
        ("backups/marine_run.zip", {"yolo11s_1280": "yolo11s_training"})]
KEEP = re.compile(r"(confusion_matrix(_normalized)?|[A-Za-z]*PR_curve|[A-Za-z]*F1_curve|results)\.png$")
LIMIT = 3_000_000
copied = 0
for zpath, mapping in SPEC:
    p = pathlib.Path(zpath)
    if not p.exists():
        print(f"  {zpath}: not found (skipped)")
        continue
    with zipfile.ZipFile(p) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if len(parts) != 2 or parts[0] not in mapping or not KEEP.search(parts[1]):
                continue
            if info.file_size > LIMIT:
                print(f"  skipped (over 3 MB): {info.filename}")
                continue
            dst = OUT / f"{mapping[parts[0]]}_{parts[1]}"
            dst.write_bytes(z.read(info))
            print(f"  {dst} ({info.file_size // 1024} KB)")
            copied += 1
print(f"staged {copied} plot files into {OUT}")
