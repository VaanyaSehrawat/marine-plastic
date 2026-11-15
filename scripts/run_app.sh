#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
JH="$ROOT/.tools/jdk-21/Contents/Home"; JAVAC="$JH/bin/javac"; JAVA="$JH/bin/java"
JAR="$ROOT/java/lib/onnxruntime-1.20.0.jar"
MODEL="$ROOT/models/yolo11s_1280/yolo11s_1280_best.onnx"
FIX="$ROOT/models/yolo11s_1280/fixture"
HTML="$ROOT/java/app/src/main/resources/dashboard.html"
OUT="$ROOT/java/out"; MODE="${1:-run}"
for f in "$JAVAC" "$JAR" "$MODEL" "$FIX/fixture.json" "$HTML"; do [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }; done
mkdir -p "$OUT" build
"$JAVAC" -encoding UTF-8 -cp "$JAR" -d "$OUT" java/app/src/main/java/mp/app/detect/*.java java/app/src/main/java/mp/app/planner/*.java java/app/src/main/java/mp/app/route/*.java java/app/src/main/java/mp/app/server/*.java
echo "COMPILED OK"
CP="$OUT:$JAR"
"$JAVA" -cp "$CP" mp.app.planner.PlannerTest
if [ "$MODE" = "--selftest" ]; then
  P=8099
  "$JAVA" -Xmx2g -cp "$CP" mp.app.server.DashboardServer "$MODEL" "$FIX" "$HTML" $P > build/selftest_server.log 2>&1 &
  PID=$!; trap 'kill $PID 2>/dev/null || true' EXIT
  for i in $(seq 1 40); do curl -sf "http://127.0.0.1:$P/api/samples" > /dev/null && break; sleep 1; done
  echo "Samples: $(curl -s "http://127.0.0.1:$P/api/samples")"
  echo "HTML starts: $(curl -s "http://127.0.0.1:$P/" | head -c 60)"
  R=$(curl -s -X POST "http://127.0.0.1:$P/api/analyze?sample=003278.png&conf=0.25&rows=6&cols=8&thr=1.0")
  echo "$R" | grep -o '"nDet":[0-9]*'
  echo "$R" | grep -o '"metrics":{[^}]*}'
  if echo "$R" | grep -q '"nDet":15,' && echo "$R" | grep -q '"route":\[' && echo "$R" | grep -q '"metrics":{'; then
    echo "SELFTEST PASSED"
  else
    echo "SELFTEST FAILED: unexpected analyze response (expected nDet=15 for 003278.png)"; echo "$R" | head -c 600; tail -20 build/selftest_server.log; exit 1
  fi
  exit 0
fi
( sleep 4; open "http://127.0.0.1:8080" >/dev/null 2>&1 || true ) &
exec "$JAVA" -Xmx2g -cp "$CP" mp.app.server.DashboardServer "$MODEL" "$FIX" "$HTML" 8080
