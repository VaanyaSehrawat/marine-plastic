#!/usr/bin/env bash
# scripts/collect_diagnostics.sh -- gather everything needed to debug a failure into ONE text file.
# Read-only except for writing logs/diagnostics-<time>.txt. Contains no passwords/keys; review before sharing.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs
OUT="logs/diagnostics-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "== diagnostics $(date) =="; echo "folder: $ROOT"
  echo "-- system --"; uname -a; command -v sw_vers >/dev/null && sw_vers; echo "arm64 capable: $(sysctl -n hw.optional.arm64 2>/dev/null)"
  echo "-- shell tools --"; bash --version | head -1; git --version; command -v java && java -version 2>&1 | head -1
  echo "-- project toolchain --"
  [ -x .tools/jdk-21/Contents/Home/bin/java ] && .tools/jdk-21/Contents/Home/bin/java -version 2>&1 | head -1
  [ -x .tools/bin/uv ] && .tools/bin/uv --version
  [ -x .venv/bin/python ] && .venv/bin/python --version
  echo "-- java/lib --"; ls -la java/lib 2>/dev/null; command -v shasum >/dev/null && shasum -a 256 java/lib/*.jar 2>/dev/null
  echo "-- git --"; git rev-parse --is-inside-work-tree 2>&1; git log --oneline 2>&1 | head -5; git status --short 2>&1 | head -20
  echo "-- disk --"; df -h . | tail -1
  echo "-- newest setup log (last 80 lines) --"; f="$(ls -t logs/setup-*.log 2>/dev/null | head -1)"; [ -n "$f" ] && { echo "$f"; tail -80 "$f"; }
  echo "-- env report --"; cat logs/env_report.txt 2>/dev/null | head -60
  echo "-- smoke artifacts --"; ls -la build/smoke 2>/dev/null; cat build/smoke/meta.properties 2>/dev/null
} > "$OUT" 2>&1
echo "wrote $ROOT/$OUT  (send this file to ChatGPT/Claude)"
