#!/usr/bin/env bash
# Step 3: all-or-nothing dashboard patch (learned scoring), run_dashboard.sh, self-tests.
# Place in ~/Projects/marine-plastic/scripts/ and run: bash scripts/integrate_step3.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
rm -f build/step3.ok
echo "[1/6] checking prerequisites"
[ -f build/step2.ok ] || { echo "STEP 3 BLOCKED: Step 2 has not passed (build/step2.ok missing)"; exit 1; }
for f in java/app/src/main/java/mp/app/route/RouteScorer.java java/app/src/main/java/mp/app/planner/CleanupPlanner.java \
         java/app/src/main/java/mp/app/server/DashboardServer.java java/app/src/main/resources/dashboard.html scripts/run_app.sh \
         models/route/route_model.onnx models/yolo11m_1280/yolo11m_1280_best.onnx models/yolo11m_1280/fixture/fixture.json; do
  [ -s "$f" ] || { echo "STEP 3 BLOCKED: missing $f"; exit 1; }
done
PY=".tools/pybin/python3.12"; [ -x "$PY" ] || PY="python3"
echo "  python: $($PY --version 2>&1)"

echo "[2/6] backing up the 4 files to be patched"
B=build/backup_pre_route
if [ ! -d "$B" ]; then
  mkdir -p "$B"
  cp java/app/src/main/java/mp/app/planner/CleanupPlanner.java java/app/src/main/java/mp/app/server/DashboardServer.java \
     java/app/src/main/resources/dashboard.html scripts/run_app.sh "$B"/ || { echo "STEP 3 FAILED: backup copy failed"; rm -rf "$B"; exit 1; }
  echo "  backed up 4 files to $B"
else
  echo "  backup already exists in $B (kept)"
fi

echo "[3/6] writing the patcher"
mkdir -p build
cat > build/patch_route.py <<'PYEOF'
import pathlib, sys

PL = "java/app/src/main/java/mp/app/planner/CleanupPlanner.java"
SV = "java/app/src/main/java/mp/app/server/DashboardServer.java"
HT = "java/app/src/main/resources/dashboard.html"
RA = "scripts/run_app.sh"

EDITS = {
PL: [
 ("planFromScores",
  "    private static double d(int[] a, int[] b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }",
  """    /** Externally supplied tile scores (for example from the learned route model). cnt is only used for display. */
    public static Plan planFromScores(double[][] sc, int[][] cnt, int rows, int cols, double threshold) {
        List<int[]> sel = new ArrayList<>();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (sc[r][c] >= threshold) sel.add(new int[] {r, c});
        List<int[]> route = twoOpt(nearestNeighbour(sel));
        return new Plan(rows, cols, sc, cnt, route, length(route), length(serpentine(sel)), sel.size());
    }

    private static double d(int[] a, int[] b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }"""),
],
SV: [
 ("import mp.app.route.RouteScorer;",
  "import mp.app.planner.CleanupPlanner;",
  "import mp.app.planner.CleanupPlanner;\nimport mp.app.route.RouteScorer;"),
 ("RouteScorer router;",
  "    private static YoloOnnxDetector detector;",
  "    private static YoloOnnxDetector detector;\n    private static RouteScorer router;"),
 ("router = new RouteScorer",
  "        detector = new YoloOnnxDetector(model.toString(), names, 0.10f, 0.7f, 400);",
  """        detector = new YoloOnnxDetector(model.toString(), names, 0.10f, 0.7f, 400);
        if (a.length > 4 && Files.exists(Path.of(a[4]))) {
            router = new RouteScorer(a[4]);
            System.out.println("Route transformer loaded: " + a[4]);
        }"""),
 ("planFromScores",
  "        CleanupPlanner.Plan plan = CleanupPlanner.plan(dets, w, h, rows, cols, thr);",
  """        CleanupPlanner.Plan plan = CleanupPlanner.plan(dets, w, h, rows, cols, thr);
        String mode = "heuristic";
        if ("learned".equals(q.get("scoring")) && router != null && rows == RouteScorer.R && cols == RouteScorer.C) {
            float[] pr;
            synchronized (router) { pr = router.priority(RouteScorer.features(r.detections(), w, h, 0.10f)); }
            double[][] sc = new double[rows][cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) sc[i][j] = Math.expm1(Math.max(0.0, pr[i * cols + j]));
            plan = CleanupPlanner.planFromScores(sc, plan.count(), rows, cols, thr);
            mode = "learned";
        }"""),
 ("append(mode)",
  r'''.append(",\"nDet\":").append(dets.size()).append(",\"detections\":[");''',
  r'''.append(",\"nDet\":").append(dets.size()).append(",\"scoring\":\"").append(mode).append("\",\"detections\":[");'''),
],
HT: [
 ('id="scoring"',
  '<label class="chk"><input type="checkbox" id="lBoxes" checked> boxes</label>',
  '<label>Scoring<select id="scoring"><option value="heuristic">Heuristic (class weights)</option><option value="learned">Learned (transformer)</option></select></label>\n<label class="chk"><input type="checkbox" id="lBoxes" checked> boxes</label>'),
 ('scoring: $("scoring")',
  'thr: $("thr").value});',
  'thr: $("thr").value, scoring: $("scoring").value});'),
 ('id="mScore"',
  '<div class="k"><span>Tiles to visit</span><b id="mSel">-</b></div>',
  '<div class="k"><span>Scoring</span><b id="mScore">-</b></div>\n<div class="k"><span>Tiles to visit</span><b id="mSel">-</b></div>'),
 ('mScore").textContent',
  '$("mSel").textContent = m.selected',
  '$("mScore").textContent = last.scoring === "learned" ? "learned (transformer)" : "heuristic";\n  $("mSel").textContent = m.selected'),
 ('last.scoring === "learned" || g.count',
  'ctx.strokeStyle = v >= thr && g.count[r][c] > 0 ?',
  'ctx.strokeStyle = v >= thr && (last.scoring === "learned" || g.count[r][c] > 0) ?'),
 ('ctx.lineWidth = v >= thr && (last.scoring',
  'ctx.lineWidth = v >= thr && g.count[r][c] > 0 ?',
  'ctx.lineWidth = v >= thr && (last.scoring === "learned" || g.count[r][c] > 0) ?'),
 ('YOLO11 (1280 px)',
  'YOLO11s (1280 px) running',
  'YOLO11 (1280 px) running'),
],
RA: [
 ("mp/app/route/*.java",
  "java/app/src/main/java/mp/app/planner/*.java",
  "java/app/src/main/java/mp/app/planner/*.java java/app/src/main/java/mp/app/route/*.java"),
],
}

results = {}
for path, edits in EDITS.items():
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"PATCH FAILED: {path} not found")
    s = p.read_text()
    n = 0
    for marker, old, new in edits:
        if marker in s:
            continue
        c = s.count(old)
        if c != 1:
            sys.exit(f"PATCH FAILED: anchor found {c} times in {path}: {old[:90]!r}")
        s = s.replace(old, new)
        n += 1
    results[path] = (p, s, n)
for path, (p, s, n) in results.items():
    if n:
        p.write_text(s)
    print(f"  {path}: {n} edit(s) applied" + ("" if n else " (already patched)"))
print("PATCHED")
PYEOF
"$PY" -m py_compile build/patch_route.py || { echo "STEP 3 FAILED: patcher has a syntax error"; exit 1; }

echo "[4/6] applying the patch (all-or-nothing)"
"$PY" build/patch_route.py || { echo "STEP 3: FAIL (patch refused, nothing was written)"; exit 1; }

echo "[5/6] writing scripts/run_dashboard.sh"
cat > scripts/run_dashboard.sh <<'REOF'
#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 1
JH="$ROOT/.tools/jdk-21/Contents/Home"; JAVAC="$JH/bin/javac"; JAVA="$JH/bin/java"
JAR="$ROOT/java/lib/onnxruntime-1.20.0.jar"
PY="$ROOT/.tools/pybin/python3.12"; [ -x "$PY" ] || PY=python3
MD="${MODEL_DIR:-yolo11m_1280}"
MODEL="$ROOT/models/$MD/${MD}_best.onnx"; FIX="$ROOT/models/$MD/fixture"
ROUTE="$ROOT/models/route/route_model.onnx"; HTML="$ROOT/java/app/src/main/resources/dashboard.html"
OUT="$ROOT/java/out"; MODE="${1:-run}"; J=java/app/src/main/java/mp/app
for f in "$JAVAC" "$JAR" "$MODEL" "$FIX/fixture.json" "$ROUTE" "$HTML"; do [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }; done
mkdir -p "$OUT" build
"$JAVAC" -encoding UTF-8 -cp "$JAR" -d "$OUT" $J/detect/*.java $J/route/*.java $J/planner/*.java $J/server/*.java || { echo "COMPILE FAILED"; exit 1; }
echo "COMPILED OK (detector: $MD)"
CP="$OUT:$JAR"
"$JAVA" -cp "$CP" mp.app.planner.PlannerTest || exit 1
if [ "$MODE" = "--selftest" ]; then
  P=8099
  EXP=$("$PY" -c "import json;d=json.load(open('$FIX/fixture.json'));print([i['n_detections'] for i in d['images'] if i['file']=='003278.png'][0])")
  "$JAVA" -Xmx3g -cp "$CP" mp.app.server.DashboardServer "$MODEL" "$FIX" "$HTML" $P "$ROUTE" > build/selftest_server.log 2>&1 &
  PID=$!; trap 'kill $PID 2>/dev/null' EXIT
  for i in $(seq 1 90); do curl -sf "http://127.0.0.1:$P/api/samples" > /dev/null && break; sleep 1; done
  Q="sample=003278.png&conf=0.25&rows=6&cols=8&thr=1.0"
  H=$(curl -s -X POST "http://127.0.0.1:$P/api/analyze?$Q")
  L=$(curl -s -X POST "http://127.0.0.1:$P/api/analyze?$Q&scoring=learned")
  L2=$(curl -s -X POST "http://127.0.0.1:$P/api/analyze?sample=003278.png&conf=0.25&rows=7&cols=9&thr=1.0&scoring=learned")
  echo "heuristic: $(echo "$H" | grep -o '"nDet":[0-9]*,"scoring":"[a-z]*"') | $(echo "$H" | grep -o '"metrics":{[^}]*}')"
  echo "learned:   $(echo "$L" | grep -o '"nDet":[0-9]*,"scoring":"[a-z]*"') | $(echo "$L" | grep -o '"metrics":{[^}]*}')"
  echo "learned on a 7x9 grid (no model for that grid): $(echo "$L2" | grep -o '"scoring":"[a-z]*"')"
  echo "route model loaded in server log: $(grep -c 'Route transformer loaded' build/selftest_server.log)"
  echo "expected nDet from fixture: $EXP"
  if [ -n "$EXP" ] && echo "$H" | grep -q "\"nDet\":$EXP,\"scoring\":\"heuristic\"" \
     && echo "$L" | grep -q '"scoring":"learned"' && echo "$L" | grep -q '"route":\[' \
     && echo "$L2" | grep -q '"scoring":"heuristic"' && [ "$(grep -c 'Route transformer loaded' build/selftest_server.log)" = "1" ]; then
    echo "SELFTEST PASSED"; exit 0
  fi
  echo "SELFTEST FAILED"; echo "$H" | head -c 300; echo; echo "$L" | head -c 300; echo; tail -20 build/selftest_server.log; exit 1
fi
lsof -ti tcp:8080 | xargs kill 2>/dev/null
( sleep 6; open "http://127.0.0.1:8080" >/dev/null 2>&1 ) &
exec "$JAVA" -Xmx3g -cp "$CP" mp.app.server.DashboardServer "$MODEL" "$FIX" "$HTML" 8080 "$ROUTE"
REOF
chmod +x scripts/run_dashboard.sh
[ -s scripts/run_dashboard.sh ] || { echo "STEP 3 FAILED: run_dashboard.sh was not written"; exit 1; }

echo "[6/6] self-tests"
echo "== new dashboard self-test (YOLO11m + route transformer) =="
bash scripts/run_dashboard.sh --selftest; rc1=$?
echo "== regression: previous self-test (YOLO11s, heuristic) =="
bash scripts/run_app.sh --selftest; rc2=$?
if [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]; then
  touch build/step3.ok; echo; echo "STEP 3: PASS"
else
  echo; echo "STEP 3: FAIL (new self-test exit $rc1, regression exit $rc2). Originals are in build/backup_pre_route/"
fi
