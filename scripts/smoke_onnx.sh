#!/usr/bin/env bash
# scripts/smoke_onnx.sh -- Milestone 2: export -> ONNX Runtime (Python) -> ONNX Runtime (Java) numerical parity.
# The Java part uses the REAL ONNX Runtime jar from java/lib (SHA-256 verified); there is no stub in this repo.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/lib.sh"; . "$ROOT/config/toolchain.env"; . "$ROOT/env.sh"
cd "$ROOT"

have java && have javac || die "JDK not found. Run: bash setup.sh"
have python || die "Python venv not found. Run: bash setup.sh"
bash scripts/fetch_java_deps.sh      # verifies pin + natives every time (fast when already present)

info "step 1/3  Python: build model, export to ONNX, check with ONNX Runtime (Python)"
python -m ml.smoke.export_smoke_model --out-dir build/smoke ${MP_SMOKE_BUILDER:+--builder "$MP_SMOKE_BUILDER"}

info "step 2/3  Java: compile (generated output only: java/out is rebuilt each run)"
rm -rf java/out && mkdir -p java/out
# shellcheck disable=SC2046
javac --release 21 -Xlint:all -cp "java/lib/*" -d java/out $(find java/app/src/main/java -name '*.java')
ok "compiled"

info "step 3/3  Java: run ONNX Runtime Java and compare with the Python reference"
ARCH_ARG=""
[ "$(uname -s)" = "Darwin" ] && ARCH_ARG="--expect-arch aarch64"
# shellcheck disable=SC2086
java -cp "java/out:java/lib/*" mp.app.smoke.OnnxSmokeTest build/smoke $ARCH_ARG
ok "ONNX smoke test PASSED (Python and Java agree within tolerance)"
