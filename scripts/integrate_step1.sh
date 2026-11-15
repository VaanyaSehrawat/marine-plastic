#!/usr/bin/env bash
# Step 1: diagnose the failed integration and stage the archives. No "set -e": every check reports its own result.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p build logs backups models/yolo11m_1280/fixture models/route
rm -f build/step1.ok
PY=".tools/pybin/python3.12"; [ -x "$PY" ] || PY="python3"
echo "== environment =="
echo "login shell: $SHELL | this script: bash $BASH_VERSION | python: $($PY --version 2>&1)"
echo "unzip: $(unzip -v 2>&1 | head -1)"
echo "== archives (exit code 0 clean, 1 warnings only, 2+ errors) =="
for z in java_fixture_m route_bundle generative_bundle; do
  f=""; for c in "backups/$z.zip" "$HOME/Downloads/$z.zip"; do [ -f "$c" ] && { f="$c"; break; }; done
  if [ -z "$f" ]; then echo "  $z.zip: NOT FOUND in backups/ or ~/Downloads"; continue; fi
  [ "$f" = "backups/$z.zip" ] || cp "$f" "backups/$z.zip"
  unzip -tq "backups/$z.zip" > build/_unzip_test.txt 2>&1; rc=$?
  echo "  $z.zip: $(du -h "backups/$z.zip" | cut -f1) | unzip -t exit code $rc | $(tail -1 build/_unzip_test.txt)"
done
echo "== replay of the original script's first lines with tracing (the last '+' line is where it stopped) =="
( set -euxo pipefail
  find_zip(){ for f in "$HOME/Downloads/$1" "backups/$1"; do [ -f "$f" ] && { echo "$f"; return 0; }; done; return 1; }
  FZ=$(find_zip java_fixture_m.zip) || { echo "MISSING java_fixture_m.zip"; exit 1; }
  RZ=$(find_zip route_bundle.zip) || { echo "MISSING route_bundle.zip"; exit 1; }
  [ "$FZ" = "backups/java_fixture_m.zip" ] || cp "$FZ" backups/
  [ "$RZ" = "backups/route_bundle.zip" ] || cp "$RZ" backups/
  unzip -o backups/java_fixture_m.zip -d models/yolo11m_1280/fixture > /dev/null
  unzip -o backups/route_bundle.zip -d models/route > /dev/null
  ls models/yolo11m_1280 models/yolo11m_1280/fixture models/route > /dev/null
) > build/_replay.txt 2>&1
rc=$?
tail -12 build/_replay.txt | sed 's/^/  /'
if [ $rc -eq 0 ]; then echo "  replay exit code 0: the original staging lines run clean here"; else echo "  replay exit code $rc: the last traced line above is the failure"; fi
echo "== staging (exit codes reported; files verified below) =="
unzip -oq backups/java_fixture_m.zip -d models/yolo11m_1280/fixture; echo "  fixture unzip exit code $?"
unzip -oq backups/route_bundle.zip -d models/route; echo "  route unzip exit code $?"
MISS=0
need(){ if [ -s "$1" ]; then echo "  OK      $1 ($(du -h "$1" | cut -f1))"; else echo "  MISSING $1"; MISS=$((MISS+1)); fi; }
echo "== required files =="
for f in .tools/jdk-21/Contents/Home/bin/javac java/lib/onnxruntime-1.20.0.jar \
  models/yolo11m_1280/yolo11m_1280_best.onnx models/yolo11m_1280/model_info.json \
  models/yolo11m_1280/fixture/fixture.json models/yolo11m_1280/fixture/001075.png models/yolo11m_1280/fixture/002907.png models/yolo11m_1280/fixture/003278.png \
  models/route/route_model.onnx models/route/route_fixture.json models/route/route_model_info.json models/route/route_data.json.gz \
  java/app/src/main/java/mp/app/detect/YoloOnnxDetector.java java/app/src/main/java/mp/app/planner/CleanupPlanner.java \
  java/app/src/main/java/mp/app/server/DashboardServer.java java/app/src/main/resources/dashboard.html scripts/run_app.sh; do need "$f"; done
echo "== content checks =="
"$PY" - <<'PYEOF'
import json
f = json.load(open("models/yolo11m_1280/fixture/fixture.json"))
print("  YOLO11m fixture:", f.get("model"), "| images:", [(i["file"], i["n_detections"]) for i in f["images"]])
r = json.load(open("models/route/route_fixture.json"))
print("  route fixture (file, tiles, features, priorities, detections):", [(e["file"], len(e["tiles"]), len(e["tiles"][0]), len(e["priority"]), len(e["dets"])) for e in r])
m = json.load(open("models/route/route_model_info.json"))
print("  route model:", m["input"], "->", m["output"])
PYEOF
echo "  python content check exit code $?"
echo "== integration state =="
for f in java/app/src/main/java/mp/app/route/RouteScorer.java java/app/src/main/java/mp/app/route/RouteTest.java scripts/test_route.sh scripts/run_dashboard.sh; do
  if [ -e "$f" ]; then echo "  exists  $f"; else echo "  absent  $f"; fi
done
echo "  git HEAD: $(git log --oneline | head -1)"
echo "  changed paths: $(git status --short | wc -l | tr -d ' ')"
if [ "$MISS" -eq 0 ]; then touch build/step1.ok; echo; echo "STEP 1: READY"; else echo; echo "STEP 1: BLOCKED ($MISS missing file(s) above)"; fi
