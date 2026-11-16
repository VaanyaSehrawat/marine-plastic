#!/usr/bin/env bash
# Detector error analysis on the Mac (no GPU, no retraining). Place in ~/Projects/marine-plastic/scripts/ and run:
#   bash scripts/detector_analysis.sh
# Uses models/route/route_data.json.gz (YOLO11m detections at conf >= 0.10 plus ground truth for val and test)
# and, if present, backups/marine_run2.zip / backups/marine_run.zip for the evaluation plots made on Kaggle.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p results/detectors/plots docs build logs scripts
[ -s models/route/route_data.json.gz ] || { echo "ANALYSIS BLOCKED: missing models/route/route_data.json.gz"; exit 1; }

PY=""
for c in .venv/bin/python .tools/pybin/python3.12 python3; do
  if [ -x "$c" ] || command -v "$c" > /dev/null 2>&1; then
    if "$c" -c "import numpy" > /dev/null 2>&1; then PY="$c"; break; fi
  fi
done
[ -n "$PY" ] || { echo "ANALYSIS BLOCKED: no Python with numpy found"; exit 1; }
echo "python: $PY ($("$PY" --version 2>&1)) | numpy $("$PY" -c 'import numpy; print(numpy.__version__)')"

cat > scripts/detector_error_analysis.py <<'PYEOF'
import gzip, json, pathlib, sys, time
import numpy as np

SKIP_GATE = "--skip-gate" in sys.argv   # test hook only; the shell wrapper never passes it
t0 = time.time()
NAMES = ["pet_bottle", "other_bottle", "plastic_bag", "box_shaped_case", "other_container", "rope", "other_string",
         "fishing_net", "buoy", "other_fishing_gear", "styrene_foam", "others", "fragment"]
EXPECT = {"val": 24643, "test": 22040}
OP, LOW, IOU = 0.25, 0.10, 0.5
BUCKETS = [("small (< 32x32 px)", 0, 32 ** 2), ("medium (32x32 to 96x96 px)", 32 ** 2, 96 ** 2), ("large (>= 96x96 px)", 96 ** 2, 1e18)]

with gzip.open("models/route/route_data.json.gz", "rt") as f:
    DATA = json.load(f)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def ap_voc(conf, tp, npos):
    if npos == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-np.asarray(conf), kind="stable")
    tp = np.asarray(tp, dtype=float)[o]; ctp = np.cumsum(tp); cfp = np.cumsum(1.0 - tp)
    rec = ctp / npos; prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def analyse(split):
    gt_rows = []                                   # per GT: class, area, recall@OP (class-aware), found@OP, right class@OP, found@LOW
    det_conf, det_tp, det_cls = [], [], []
    for rec in DATA[split]:
        d = np.array(rec["dets"], dtype=np.float64).reshape(-1, 6)
        g = np.array(rec["gt"], dtype=np.float64).reshape(-1, 5)
        d = d[d[:, 4] >= LOW]
        d = d[np.argsort(-d[:, 4], kind="stable")]
        M = iou_mat(d[:, :4], g[:, :4])
        matched_conf = np.full(len(g), -1.0)
        tp = np.zeros(len(d), dtype=bool)
        for i in range(len(d)):                    # VOC-style greedy matching, class-aware, highest confidence first
            same = np.where(g[:, 4] == d[i, 5])[0]
            if len(same) == 0:
                continue
            j = same[np.argmax(M[i, same])]
            if M[i, j] >= IOU and matched_conf[j] < 0:
                matched_conf[j] = d[i, 4]; tp[i] = True
        det_conf.append(d[:, 4]); det_tp.append(tp); det_cls.append(d[:, 5].astype(int))
        for j in range(len(g)):
            area = (g[j, 2] - g[j, 0]) * (g[j, 3] - g[j, 1])
            hi = d[:, 4] >= OP
            ious_hi = M[hi, j] if len(d) else np.zeros(0)
            found_op = bool(len(ious_hi) and ious_hi.max() >= IOU)
            right = False
            if found_op:
                cand = np.where(hi)[0][ious_hi >= IOU]
                best = cand[np.argmax(M[cand, j])]
                right = bool(d[best, 5] == g[j, 4])
            found_low = bool(len(d) and M[:, j].max() >= IOU)
            gt_rows.append((int(g[j, 4]), float(area), bool(matched_conf[j] >= OP), found_op, right, found_low))
    G = np.array(gt_rows, dtype=np.float64).reshape(-1, 6)
    conf = np.concatenate(det_conf) if det_conf else np.zeros(0)
    tp = np.concatenate(det_tp) if det_tp else np.zeros(0, dtype=bool)
    cls = np.concatenate(det_cls) if det_cls else np.zeros(0, dtype=int)
    return G, conf, tp, cls


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def fmt(v, d=1):
    return "n/a" if v != v else f"{v:.{d}f}"


out, md = {}, []
w = md.append
w("# Detector error analysis (generated by scripts/detector_error_analysis.py; do not edit by hand)\n")
w("Model: YOLO11m (the dashboard detector), 1280 px letterbox input, NMS IoU 0.7, detections kept at confidence 0.10 and above. A detection is a true positive when its IoU with a not-yet-matched ground-truth box of the same class is at least 0.5 (highest confidence first). Operating point for recall and precision: confidence 0.25, the dashboard default.\n")
w("These numbers come from our own matching code on the saved detections. They are not the Ultralytics mAP numbers in docs/RESULTS.md (those use confidence 0.001, NMS IoU 0.6 and 101-point interpolation), so AP50 here is expected to differ from them.\n")
for split in ("test", "val"):
    G, conf, tp, cls = analyse(split)
    if len(G) != EXPECT[split] and not SKIP_GATE:
        sys.exit(f"GATE: FAIL ({split} has {len(G)} ground-truth boxes, expected {EXPECT[split]})")
    n = len(G); op = conf >= OP
    res = {"n_gt": n, "n_dets_op": int(op.sum()), "recall_class_aware_op": pct(G[:, 2].sum(), n), "found_any_class_op": pct(G[:, 3].sum(), n),
           "class_correct_among_found_op": pct(G[:, 4].sum(), G[:, 3].sum()), "found_any_class_low": pct(G[:, 5].sum(), n),
           "precision_op": pct(tp[op].sum(), op.sum()), "per_class": {}, "per_size": {}}
    aps = []
    for k, name in enumerate(NAMES):
        sel = G[:, 0] == k; dk = cls == k; npos = int(sel.sum())
        ap = ap_voc(conf[dk], tp[dk], npos); aps.append(ap)
        dop = dk & op
        res["per_class"][name] = {"n_gt": npos, "ap50": ap, "recall_op": pct(G[sel, 2].sum(), npos), "precision_op": pct(tp[dop].sum(), dop.sum()),
                                  "found_any_class_op": pct(G[sel, 3].sum(), npos), "class_correct_among_found_op": pct(G[sel, 4].sum(), G[sel, 3].sum())}
    res["map50_own"] = float(np.nanmean(aps))
    for label, lo, hi in BUCKETS:
        sel = (G[:, 1] >= lo) & (G[:, 1] < hi); m = int(sel.sum())
        res["per_size"][label] = {"n_gt": m, "share_pct": pct(m, n), "recall_op": pct(G[sel, 2].sum(), m), "found_any_class_op": pct(G[sel, 3].sum(), m),
                                  "found_any_class_low": pct(G[sel, 5].sum(), m)}
    out[split] = res
    w(f"## {'Test' if split == 'test' else 'Validation'} split ({n} ground-truth boxes, {len(DATA[split])} images)\n")
    w(f"Own mAP50 over the 13 classes: {res['map50_own']:.3f}. At confidence 0.25: {res['n_dets_op']} detections, precision {fmt(res['precision_op'])}%, class-aware recall {fmt(res['recall_class_aware_op'])}%.")
    w(f"Ground-truth boxes found by any detection (IoU 0.5, class ignored): {fmt(res['found_any_class_op'])}% at confidence 0.25 and {fmt(res['found_any_class_low'])}% at 0.10. Among the boxes found at 0.25, the best-overlapping detection has the right class in {fmt(res['class_correct_among_found_op'])}% of cases.\n")
    w("### By object size (ground-truth box area in original image pixels)\n")
    w("| size | boxes | share of all boxes (%) | class-aware recall at 0.25 (%) | found by any class at 0.25 (%) | found by any class at 0.10 (%) |")
    w("|---|---:|---:|---:|---:|---:|")
    for label, _, _ in BUCKETS:
        r = res["per_size"][label]
        w(f"| {label} | {r['n_gt']} | {fmt(r['share_pct'])} | {fmt(r['recall_op'])} | {fmt(r['found_any_class_op'])} | {fmt(r['found_any_class_low'])} |")
    w("\n### By class\n")
    w("| class | boxes | AP50 (own) | recall at 0.25 (%) | precision at 0.25 (%) | found by any class at 0.25 (%) | right class among found (%) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for name in NAMES:
        r = res["per_class"][name]
        w(f"| {name} | {r['n_gt']} | {fmt(r['ap50'], 3)} | {fmt(r['recall_op'])} | {fmt(r['precision_op'])} | {fmt(r['found_any_class_op'])} | {fmt(r['class_correct_among_found_op'])} |")
    w("")

t = out["test"]
sm = t["per_size"][BUCKETS[0][0]]; lg = t["per_size"][BUCKETS[2][0]]
lost_loc = 100.0 - t["found_any_class_op"]
lost_cls = t["found_any_class_op"] - t["recall_class_aware_op"]
w("## Reading the test-split numbers\n")
w(f"- Of all ground-truth boxes, {fmt(lost_loc)}% have no detection at all at confidence 0.25 (missed or not localised), and a further {fmt(lost_cls)}% are localised but carry the wrong class or lose the match to another box; the remaining {fmt(t['recall_class_aware_op'])}% are correctly found and classified.")
w(f"- Small boxes are {fmt(sm['share_pct'])}% of the test boxes and class-aware recall on them is {fmt(sm['recall_op'])}%, against {fmt(lg['recall_op'])}% for large boxes.")
best = max(NAMES, key=lambda k: -1 if t["per_class"][k]["ap50"] != t["per_class"][k]["ap50"] else t["per_class"][k]["ap50"])
worst = min(NAMES, key=lambda k: 2 if t["per_class"][k]["ap50"] != t["per_class"][k]["ap50"] else t["per_class"][k]["ap50"])
w(f"- Highest own AP50: {best} ({t['per_class'][best]['ap50']:.3f}); lowest: {worst} ({t['per_class'][worst]['ap50']:.3f}).")
w("- These are one model, one training run and one matching rule (IoU 0.5); do not read small differences between classes as established.\n")
w("## Plausibility check\n")
ref = 0.252
status = "OK" if 0.12 <= t["map50_own"] <= 0.36 else "WARN (outside the expected range)"
w(f"Own test mAP50 {t['map50_own']:.3f} against the Ultralytics test mAP50 {ref:.3f} for the same model: {status}. A moderate gap is expected because of the different confidence floor, NMS threshold and interpolation.")
if not SKIP_GATE:
    w("Ground-truth totals match the dataset counts (val 24,643 and test 22,040).")
pathlib.Path("docs").mkdir(exist_ok=True); pathlib.Path("results").mkdir(exist_ok=True)
json.dump(out, open("results/detector_error_analysis.json", "w"), indent=1)
text = "\n".join(md) + "\n"
pathlib.Path("docs/DETECTOR_ANALYSIS.md").write_text(text)
print(text)
print(f"wrote results/detector_error_analysis.json and docs/DETECTOR_ANALYSIS.md | {time.time() - t0:.1f} s")
print("ERROR ANALYSIS: DONE")
PYEOF

cat > scripts/stage_detector_plots.py <<'PYEOF'
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
PYEOF
"$PY" -m py_compile scripts/detector_error_analysis.py scripts/stage_detector_plots.py || { echo "ANALYSIS FAILED: script syntax error"; exit 1; }

echo "== [1/3] detector error analysis =="
"$PY" scripts/detector_error_analysis.py; rc=$?
[ $rc -eq 0 ] || { echo; echo "ANALYSIS FAILED (exit $rc): nothing committed. Send me the full output."; exit 1; }

echo "== [2/3] staging evaluation plots made on Kaggle =="
"$PY" scripts/stage_detector_plots.py || { echo "PLOT STAGING FAILED"; exit 1; }

echo "== [3/3] commit and push =="
git add results/detector_error_analysis.json docs/DETECTOR_ANALYSIS.md results/detectors scripts/detector_error_analysis.py scripts/stage_detector_plots.py scripts/detector_analysis.sh
BAD=0
for f in $(git diff --cached --name-only); do
  case "$f" in *.pt|*.pth|*.onnx|*.zip|*.tar.gz) echo "  FORBIDDEN staged file: $f"; BAD=1;; esac
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 5000000 ]; then echo "  FILE OVER 5 MB staged: $f"; BAD=1; fi
done
if [ $BAD -ne 0 ]; then git reset -q; echo "FAILED: unstaged everything, nothing committed."; exit 1; fi
echo "  staged files: $(git diff --cached --name-only | wc -l | tr -d ' ')"
if git diff --cached --quiet; then
  echo "  nothing new to commit"
else
  git commit -q -m "docs(m7): detector error analysis (per class, size buckets, missed vs misclassified) and staged evaluation plots" || { echo "FAILED: commit failed"; exit 1; }
  echo "  committed: $(git log --oneline | head -1)"
fi
git push > build/_push_det.txt 2>&1; prc=$?
tail -3 build/_push_det.txt | sed 's/^/  /'
if [ $prc -ne 0 ]; then echo "COMMITTED LOCALLY, PUSH FAILED (exit $prc). Send me the lines above."; exit 1; fi
echo "  $(git status -sb | head -1)"
echo; echo "DETECTOR ANALYSIS: COMMITTED AND PUSHED"
