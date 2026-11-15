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
