#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 1
JH="$ROOT/.tools/jdk-21/Contents/Home"; JAVAC="$JH/bin/javac"; JAVA="$JH/bin/java"
JAR="$ROOT/java/lib/onnxruntime-1.20.0.jar"
M="$ROOT/models/yolo11m_1280/yolo11m_1280_best.onnx"; MF="$ROOT/models/yolo11m_1280/fixture"
RM="$ROOT/models/route/route_model.onnx"; RF="$ROOT/models/route/route_fixture.json"
for f in "$JAVAC" "$JAR" "$M" "$MF/fixture.json" "$ROOT/models/yolo11m_1280/model_info.json" "$RM" "$RF"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
OUT="$ROOT/java/out"; mkdir -p "$OUT"; J=java/app/src/main/java/mp/app
"$JAVAC" -encoding UTF-8 -cp "$JAR" -d "$OUT" $J/detect/*.java $J/route/*.java $J/planner/*.java $J/server/*.java || { echo "COMPILE FAILED"; exit 1; }
echo "COMPILED OK"
CP="$OUT:$JAR"
echo "== YOLO11m fixture test =="
"$JAVA" -Xmx3g -Djava.awt.headless=true -cp "$CP" mp.app.detect.FixtureTest "$M" "$MF" || exit 1
echo "== Route transformer test =="
"$JAVA" -Xmx3g -Djava.awt.headless=true -cp "$CP" mp.app.route.RouteTest "$RM" "$RF" "$M" "$MF" || exit 1
