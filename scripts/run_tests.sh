#!/usr/bin/env bash
# scripts/run_tests.sh -- the single reproducible test command.   bash scripts/run_tests.sh [--python-only]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/lib.sh"; . "$ROOT/config/toolchain.env"; . "$ROOT/env.sh"
cd "$ROOT"

stage "Python unit tests (pytest)"
python -m pytest tests/python -q

if [ "${1:-}" != "--python-only" ]; then
  stage "ONNX integration smoke test (Python -> Java)"
  bash scripts/smoke_onnx.sh
  # Later milestones add: Java fundamentals tests, data-leakage checks, planner tests.
fi
ok "all tests passed"
