#!/usr/bin/env bash
# Read-only: changes no file.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=".tools/pybin/python3.12"; [ -x "$PY" ] || PY="python3"
BAD=0
echo "== markers from earlier steps =="
for f in build/step1.ok build/step2.ok; do [ -e "$f" ] && echo "  OK      $f" || { echo "  MISSING $f"; BAD=1; }; done
echo "== route files =="
for f in java/app/src/main/java/mp/app/route/RouteScorer.java java/app/src/main/java/mp/app/route/RouteTest.java scripts/test_route.sh \
         models/route/route_model.onnx models/route/route_fixture.json models/yolo11m_1280/yolo11m_1280_best.onnx models/yolo11m_1280/fixture/fixture.json; do
  [ -s "$f" ] && echo "  OK      $f" || { echo "  MISSING $f"; BAD=1; }
done
echo "== git =="
echo "  branch: $(git status -sb | head -1)"
echo "  HEAD:   $(git log --oneline | head -1)"
echo "  uncommitted or untracked paths:"; git status --short | sed 's/^/    /'
echo "  dashboard files changed since the last commit (blank means untouched):"
git diff --stat -- java/app/src/main/java/mp/app/planner/CleanupPlanner.java java/app/src/main/java/mp/app/server/DashboardServer.java java/app/src/main/resources/dashboard.html scripts/run_app.sh | sed 's/^/    /'
echo "== patch anchors in the current files =="
"$PY" - <<'PYEOF'
import pathlib, sys
PL = "java/app/src/main/java/mp/app/planner/CleanupPlanner.java"
SV = "java/app/src/main/java/mp/app/server/DashboardServer.java"
HT = "java/app/src/main/resources/dashboard.html"
RA = "scripts/run_app.sh"
CHK = {
 PL: (["planFromScores"], ["    private static double d(int[] a, int[] b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }"]),
 SV: (["RouteScorer"], ["import mp.app.planner.CleanupPlanner;", "    private static YoloOnnxDetector detector;",
       "        detector = new YoloOnnxDetector(model.toString(), names, 0.10f, 0.7f, 400);",
       "        CleanupPlanner.Plan plan = CleanupPlanner.plan(dets, w, h, rows, cols, thr);",
       r'.append(",\"nDet\":").append(dets.size()).append(",\"detections\":[");']),
 HT: (['id="scoring"'], ['<label class="chk"><input type="checkbox" id="lBoxes" checked> boxes</label>', 'thr: $("thr").value});',
       '<div class="k"><span>Tiles to visit</span><b id="mSel">-</b></div>', '$("mSel").textContent = m.selected',
       'ctx.strokeStyle = v >= thr && g.count[r][c] > 0 ?', 'ctx.lineWidth = v >= thr && g.count[r][c] > 0 ?', 'YOLO11s (1280 px) running']),
 RA: (["mp/app/route/*.java"], ["java/app/src/main/java/mp/app/planner/*.java"]),
}
bad = 0
for path, (markers, anchors) in CHK.items():
    p = pathlib.Path(path)
    if not p.exists():
        print(f"  MISSING {path}"); bad += 1; continue
    s = p.read_text()
    if all(m in s for m in markers):
        print(f"  patched   {path} (route edits already present; Step 3 will skip them)")
    elif any(m in s for m in markers):
        print(f"  UNEXPECTED {path}: only part of the route edits are present"); bad += 1
    else:
        counts = [s.count(a) for a in anchors]
        if all(c == 1 for c in counts):
            print(f"  unpatched {path} (all {len(anchors)} anchors found exactly once)")
        else:
            print(f"  UNEXPECTED {path}: anchor counts {counts} (each must be 1)"); bad += 1
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || BAD=1
echo
if [ $BAD -eq 0 ]; then echo "PREFLIGHT OK: nothing was changed; run Step 3 (scripts/integrate_step3.sh) next"; else echo "PREFLIGHT BLOCKED: nothing was changed; send me this output"; fi
