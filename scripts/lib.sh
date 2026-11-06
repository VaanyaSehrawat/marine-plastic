# scripts/lib.sh -- shared helpers. Source it:  . "$ROOT/scripts/lib.sh"
# Written for bash 3.2 (the version macOS ships): no associative arrays, no mapfile, no ${var,,}.

if [ -t 1 ]; then
  _R=$'\033[31m'; _G=$'\033[32m'; _Y=$'\033[33m'; _B=$'\033[34m'; _N=$'\033[0m'
else
  _R=""; _G=""; _Y=""; _B=""; _N=""
fi

info()  { printf '%s[ .. ]%s %s\n' "$_B" "$_N" "$*"; }
ok()    { printf '%s[ OK ]%s %s\n' "$_G" "$_N" "$*"; }
warn()  { printf '%s[WARN]%s %s\n' "$_Y" "$_N" "$*"; }
fail()  { printf '%s[FAIL]%s %s\n' "$_R" "$_N" "$*"; }
die()   { fail "$*"; exit 1; }
stage() { printf '\n%s== %s ==%s\n' "$_B" "$*" "$_N"; }

have() { command -v "$1" >/dev/null 2>&1; }

sha256_of() {
  if have shasum; then shasum -a 256 "$1" | awk '{print $1}'; else sha256sum "$1" | awk '{print $1}'; fi
}
sha1_of() {
  if have shasum; then shasum -a 1 "$1" | awk '{print $1}'; else sha1sum "$1" | awk '{print $1}'; fi
}

# Sets MP_OS, JDK_FILE, JDK_SHA, UV_FILE, UV_SHA, JAVA_HOME_LOCAL (requires config/toolchain.env sourced and ROOT set).
select_assets() {
  MP_OS="$(uname -s)"
  case "$MP_OS/$(uname -m)" in
    Darwin/arm64)
      JDK_FILE="$JDK_FILE_MAC_ARM64"; JDK_SHA="$JDK_SHA256_MAC_ARM64"
      UV_FILE="$UV_FILE_MAC_ARM64";   UV_SHA="$UV_SHA256_MAC_ARM64"
      JAVA_HOME_LOCAL="$ROOT/.tools/jdk-21/Contents/Home" ;;
    Linux/x86_64)
      [ "${MP_DEV_ALLOW_LINUX:-0}" = "1" ] || die "Linux is not a supported target. (Developer-only override: MP_DEV_ALLOW_LINUX=1)"
      JDK_FILE="$JDK_FILE_LINUX_X64"; JDK_SHA="$JDK_SHA256_LINUX_X64"
      UV_FILE="$UV_FILE_LINUX_X64";   UV_SHA="$UV_SHA256_LINUX_X64"
      JAVA_HOME_LOCAL="$ROOT/.tools/jdk-21" ;;
    *) die "Unsupported platform: $(uname -s)/$(uname -m). This project targets macOS on Apple Silicon (arm64)." ;;
  esac
}

# download URL DEST EXPECTED_SHA256  -- resumable-safe: writes DEST.part, verifies, then renames.
download_verified() {
  local url="$1" dest="$2" want="$3" got
  if [ -f "$dest" ]; then
    got="$(sha256_of "$dest")"
    if [ "$got" = "$want" ]; then ok "cached download verified: $(basename "$dest")"; return 0; fi
    warn "cached file has wrong checksum; re-downloading $(basename "$dest")"
  fi
  mkdir -p "$(dirname "$dest")"
  info "downloading $(basename "$dest")"
  curl -fL --retry 3 --connect-timeout 20 -o "$dest.part" "$url" || { fail "download failed: $url"; return 1; }
  got="$(sha256_of "$dest.part")"
  if [ "$got" != "$want" ]; then
    fail "SHA-256 mismatch for $(basename "$dest"): expected $want, got $got"
    fail "The downloaded file was NOT used. Re-run setup once; if the mismatch repeats, stop and report it (do not bypass the check)."
    return 1
  fi
  mv "$dest.part" "$dest"
  ok "checksum verified: $(basename "$dest")"
}

# Move a path aside (never delete user-visible state).
move_aside() {
  local p="$1" ts
  ts="$(date +%Y%m%d-%H%M%S)"
  mv "$p" "$p.old-$ts"
  warn "moved existing $(basename "$p") to $(basename "$p").old-$ts"
}
