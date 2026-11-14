#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
JH="$ROOT/.tools/jdk-21/Contents/Home"; JAVAC="$JH/bin/javac"; JAVA="$JH/bin/java"
JAR="$ROOT/java/lib/onnxruntime-1.20.0.jar"
MODEL="$ROOT/models/yolo11s_1280/yolo11s_1280_best.onnx"
FIX="$ROOT/models/yolo11s_1280/fixture"
for f in "$JAVAC" "$JAR" "$MODEL" "$FIX/fixture.json" "$ROOT/models/yolo11s_1280/model_info.json"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
OUT="$ROOT/java/out"; mkdir -p "$OUT" "$ROOT/build/detect_demo"
"$JAVAC" -encoding UTF-8 -cp "$JAR" -d "$OUT" java/app/src/main/java/mp/app/detect/*.java
echo "COMPILED OK with $("$JAVA" -version 2>&1 | head -1)"
CP="$OUT:$JAR"
"$JAVA" -Xmx2g -Djava.awt.headless=true -cp "$CP" mp.app.detect.FixtureTest "$MODEL" "$FIX"
for f in "$FIX"/*.png; do
  "$JAVA" -Xmx2g -Djava.awt.headless=true -cp "$CP" mp.app.detect.DetectCli "$MODEL" "$f" "$ROOT/build/detect_demo/$(basename "$f" .png)_detected.png"
done
open "$ROOT/build/detect_demo" 2>/dev/null || true
