#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
JH="$ROOT/.tools/jdk-21/Contents/Home"; JAVAC="$JH/bin/javac"; JAVA="$JH/bin/java"
SRC="java/fundamentals/src/main/java/mp/fund"; OUT="$ROOT/java/out"
IMG="$ROOT/models/yolo11s_1280/fixture/003278.png"
for f in "$JAVAC" "$IMG"; do [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }; done
mkdir -p "$OUT" build/svd_demo
"$JAVAC" -encoding UTF-8 -d "$OUT" "$SRC"/*.java
echo "COMPILED OK"
"$JAVA" -cp "$OUT" mp.fund.FundamentalsTest
"$JAVA" -Xmx1g -cp "$OUT" mp.fund.FundamentalsDemo "$IMG" "$ROOT/build/svd_demo/svd_003278.png"
open "$ROOT/build/svd_demo" 2>/dev/null || true
