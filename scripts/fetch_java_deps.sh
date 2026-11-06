#!/usr/bin/env bash
# scripts/fetch_java_deps.sh -- download + strictly verify the ONNX Runtime Java jar in java/lib/. Idempotent.
#  * SHA-256 of the jar must equal ORT_JAR_SHA256 (config/toolchain.env). A wrong jar is never used.
#  * On macOS the jar must contain BOTH ai/onnxruntime/native/osx-aarch64/libonnxruntime.dylib and
#    ai/onnxruntime/native/osx-aarch64/libonnxruntime4j_jni.dylib (Apple Silicon natives).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/lib.sh"; . "$ROOT/config/toolchain.env"; . "$ROOT/env.sh"

JAR="$ROOT/java/lib/onnxruntime-$ORT_VERSION.jar"
URL="$ORT_MAVEN_BASE/$ORT_VERSION/onnxruntime-$ORT_VERSION.jar"
mkdir -p "$ROOT/java/lib"
[ -n "$ORT_JAR_SHA256" ] || die "ORT_JAR_SHA256 is empty in config/toolchain.env; refusing to use an unverified jar."

# Required native entries depend on the platform (override only for developer tests).
case "${MP_NATIVE_PLATFORM:-$(uname -s)/$(uname -m)}" in
  Darwin/arm64|osx-aarch64) NATIVE_DIR="osx-aarch64"; LIB_EXT="dylib" ;;
  Linux/x86_64|linux-x64)   NATIVE_DIR="linux-x64";   LIB_EXT="so" ;;
  *) die "Unsupported platform for ONNX Runtime Java natives: $(uname -s)/$(uname -m)" ;;
esac
REQ1="ai/onnxruntime/native/$NATIVE_DIR/libonnxruntime.$LIB_EXT"
REQ2="ai/onnxruntime/native/$NATIVE_DIR/libonnxruntime4j_jni.$LIB_EXT"

if [ -f "$JAR" ]; then
  if [ "$(sha256_of "$JAR")" = "$ORT_JAR_SHA256" ]; then
    ok "jar already present and matches the pinned SHA-256"
  else
    warn "existing jar does not match the pinned SHA-256"
    move_aside "$JAR"
  fi
fi

if [ ! -f "$JAR" ]; then
  info "downloading onnxruntime-$ORT_VERSION.jar from Maven Central"
  curl -fL --retry 3 --connect-timeout 20 -o "$JAR.part" "$URL" \
    || die "could not download $URL (offline, or that version is not on Maven Central)."
  got="$(sha256_of "$JAR.part")"
  if [ "$got" != "$ORT_JAR_SHA256" ]; then
    rm -f "$JAR.part"
    die "jar SHA-256 mismatch: expected $ORT_JAR_SHA256, got $got. The file was discarded and NOT used."
  fi
  mv "$JAR.part" "$JAR"
  ok "jar downloaded; SHA-256 matches the pin"
fi

have jar || die "the 'jar' tool is missing (run: bash setup.sh to install the project JDK)."
listing="$(jar tf "$JAR")"
if printf '%s\n' "$listing" | grep -Fxq "$REQ1" && printf '%s\n' "$listing" | grep -Fxq "$REQ2"; then
  ok "native libraries present: $REQ1"
  ok "native libraries present: $REQ2"
else
  fail "the jar is missing required native libraries:"
  fail "  $REQ1"
  fail "  $REQ2"
  fail "native folders actually found in the jar:"
  printf '%s\n' "$listing" | grep '^ai/onnxruntime/native/' | cut -d/ -f1-4 | sort -u | sed 's/^/       /'
  die "ONNX Runtime Java cannot load on this platform with this jar. See DECISIONS.md (fallback: HTTP detector, option B)."
fi
