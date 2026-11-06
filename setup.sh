#!/usr/bin/env bash
# setup.sh -- one-command, re-runnable setup for the Marine Plastic project (macOS, Apple Silicon).
#
#   bash setup.sh              full install (JDK 21 + Python 3.12 venv + PyTorch stack + ONNX Runtime Java) + smoke test
#   bash setup.sh --minimal    skip PyTorch/data-science packages (fast; enough for the ONNX smoke test)
#   bash setup.sh --check      only report what is installed / missing (creates only logs/ and a log file)
#   bash setup.sh --help
#
# Safety: installs ONLY inside this project folder (.tools/, .venv/, java/lib/, logs/). Never uses sudo, never
# edits ~/.zshrc or system Java/Python, never deletes anything (wrong-version leftovers are renamed *.old-<time>).
# Every download is checked against a pinned SHA-256 (or Maven Central's published SHA-1 for the jar).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs
# shellcheck disable=SC1091
. "$ROOT/scripts/lib.sh"
# shellcheck disable=SC1091
. "$ROOT/config/toolchain.env"

MINIMAL=0; CHECK_ONLY=0; SKIP_SMOKE=0; SKIP_JAVA_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --minimal) MINIMAL=1 ;;
    --check) CHECK_ONLY=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --skip-java-deps) SKIP_JAVA_DEPS=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

CURRENT_STAGE="start"
on_err() {
  [ "${BASH_SUBSHELL:-0}" -gt 0 ] || return 0     # the outer shell sees the same failure; report it only once
  fail "Setup stopped during stage: $CURRENT_STAGE (script line $1)."
  fail "Nothing was deleted. Fix the problem shown above and simply re-run:  bash setup.sh"
  fail "If you are stuck, send ChatGPT/Claude the newest file in logs/ (it contains this whole run)."
}
trap 'on_err $LINENO' ERR

# ---------------------------------------------------------------- stage 1
stage_preflight() {
  CURRENT_STAGE="1/9 preflight checks"; stage "$CURRENT_STAGE"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" != "arm64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
    die "This terminal is running under Rosetta (x86_64). Quit Terminal, untick Get Info -> 'Open using Rosetta', reopen, and re-run."
  fi
  select_assets

  if [ "$MP_OS" = "Darwin" ]; then
    [ "$(id -u)" != "0" ] || die "Do not run setup with sudo. It installs only inside this folder."
    local v major
    v="$(sw_vers -productVersion)"; major="${v%%.*}"
    [ "$major" -ge 13 ] || die "macOS $v is too old (need 13 or newer)."
    ok "macOS $v (build $(sw_vers -buildVersion)), arm64"
    if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" != "1" ]; then
      die "This Mac does not report Apple Silicon."
    fi
    xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools missing. Run: xcode-select --install  (then re-run setup)."
    ok "Xcode Command Line Tools present"
  else
    warn "DEVELOPER MODE on $MP_OS (MP_DEV_ALLOW_LINUX=1): not a supported target"
  fi

  local t
  for t in curl tar git awk sed grep df; do have "$t" || die "required tool missing: $t"; done
  have shasum || have sha256sum || die "need shasum or sha256sum"
  ok "required command-line tools present (curl, tar, git, shasum, ...)"

  case "$ROOT" in *" "*) die "Project path contains a space: '$ROOT'. Move the folder to a path without spaces." ;; esac
  case "$ROOT" in
    *"/Library/Mobile Documents/"*|*/Desktop/*|*/Documents/*)
      warn "Project is inside Desktop/Documents/iCloud. Syncing can slow or corrupt .venv; ~/Developer is safer." ;;
  esac

  local free_kb free_gb
  free_kb="$(df -k "$ROOT" | awk 'NR==2 {print $4}')"; free_gb=$((free_kb / 1024 / 1024))
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || die "Only ${free_gb} GB free; need at least ${MIN_FREE_GB} GB."
  [ "$free_gb" -ge 10 ] || warn "Only ${free_gb} GB free. Setup fits, but datasets/checkpoints later need more; free up space soon."
  ok "disk space: ${free_gb} GB free"

  if [ "$CHECK_ONLY" = "0" ]; then
    curl -fsI --connect-timeout 15 https://github.com >/dev/null 2>&1 || die "Cannot reach github.com. Check your internet connection."
    ok "network: github.com reachable"
  fi
  if [ "$MP_OS" = "Darwin" ] && have java; then
    info "system Java found ($(java -version 2>&1 | head -1)); it will be left untouched."
  fi
}

# ---------------------------------------------------------------- stage 2
stage_layout() {
  CURRENT_STAGE="2/9 project layout"; stage "$CURRENT_STAGE"
  local d
  for d in .tools/bin .tools/cache logs build data/raw data/interim data/processed data/splits models experiments reports \
           notebooks web docs config requirements scripts tests/python \
           ml/data ml/train ml/eval ml/export ml/advanced ml/smoke java/fundamentals java/app java/lib; do
    [ "$CHECK_ONLY" = "1" ] && { [ -d "$d" ] || warn "would create $d"; continue; }
    mkdir -p "$d"
  done
  for d in data/raw data/interim data/processed data/splits models experiments reports notebooks web java/lib; do
    [ "$CHECK_ONLY" = "1" ] && continue
    [ -e "$d/.gitkeep" ] || : > "$d/.gitkeep"
  done
  ok "directory structure present (existing files untouched)"
}

# ---------------------------------------------------------------- stage 3
stage_jdk() {
  CURRENT_STAGE="3/9 JDK 21 (project-local)"; stage "$CURRENT_STAGE"
  local jdir="$ROOT/.tools/jdk-21"
  if [ -x "$JAVA_HOME_LOCAL/bin/javac" ] && "$JAVA_HOME_LOCAL/bin/javac" -version 2>&1 | grep -q '^javac 21\.'; then
    ok "JDK already installed: $("$JAVA_HOME_LOCAL/bin/javac" -version 2>&1)"
  elif [ "$CHECK_ONLY" = "1" ]; then
    warn "JDK 21 not installed yet (setup would download Temurin $JDK_VERSION)"; return 0
  else
    download_verified "$JDK_BASE_URL/$JDK_TAG_URLENC/$JDK_FILE" "$ROOT/.tools/cache/$JDK_FILE" "$JDK_SHA"
    [ -e "$jdir" ] && move_aside "$jdir"
    local tmp="$ROOT/.tools/jdk-21.tmp"; rm -rf "$tmp"; mkdir -p "$tmp"
    info "extracting JDK (about 300 MB)"
    tar -xzf "$ROOT/.tools/cache/$JDK_FILE" -C "$tmp" --strip-components=1
    mv "$tmp" "$jdir"
    ok "JDK installed to .tools/jdk-21"
  fi
  "$JAVA_HOME_LOCAL/bin/java" -version 2>&1 | head -1 | sed 's/^/       /'
  "$JAVA_HOME_LOCAL/bin/javac" -version 2>&1 | sed 's/^/       /'
  if [ "$MP_OS" = "Darwin" ] && have file; then
    file "$JAVA_HOME_LOCAL/bin/java" | grep -q arm64 || die "Installed JDK is not an arm64 binary."
    ok "JDK binary is arm64 (native Apple Silicon)"
  fi
}

# ---------------------------------------------------------------- stage 4
stage_python() {
  CURRENT_STAGE="4/9 Python $PYTHON_SERIES venv (project-local, via uv)"; stage "$CURRENT_STAGE"
  # shellcheck disable=SC1091
  . "$ROOT/env.sh"
  local uvbin="$ROOT/.tools/bin/uv"
  if [ -x "$uvbin" ] && "$uvbin" --version 2>/dev/null | grep -q "$UV_VERSION"; then
    ok "uv $UV_VERSION present"
  elif [ "$CHECK_ONLY" = "1" ]; then
    warn "uv not installed yet"; return 0
  else
    download_verified "$UV_BASE_URL/$UV_VERSION/$UV_FILE" "$ROOT/.tools/cache/$UV_FILE" "$UV_SHA"
    local tmp="$ROOT/.tools/uv.tmp"; rm -rf "$tmp"; mkdir -p "$tmp"
    tar -xzf "$ROOT/.tools/cache/$UV_FILE" -C "$tmp"
    cp "$(find "$tmp" -type f -name uv | head -1)" "$uvbin"; chmod +x "$uvbin"; rm -rf "$tmp"
    ok "uv $("$uvbin" --version | awk '{print $2}') installed to .tools/bin"
  fi
  if [ "$CHECK_ONLY" = "1" ]; then
    [ -x "$ROOT/.venv/bin/python" ] && ok ".venv exists ($("$ROOT/.venv/bin/python" --version))" || warn ".venv not created yet"
    return 0
  fi
  info "ensuring a private CPython $PYTHON_SERIES (your system Python is not touched)"
  "$uvbin" python install "$PYTHON_SERIES"
  local want_ok=0
  if [ -x "$ROOT/.venv/bin/python" ]; then
    "$ROOT/.venv/bin/python" -c "import sys; sys.exit(0 if sys.version_info[:2]==tuple(map(int,'$PYTHON_SERIES'.split('.'))) else 1)" && want_ok=1
    [ "$want_ok" = "1" ] || move_aside "$ROOT/.venv"
  fi
  if [ "$want_ok" = "1" ]; then
    ok ".venv already uses Python $PYTHON_SERIES"
  else
    "$uvbin" venv --python "$PYTHON_SERIES" --seed "$ROOT/.venv"
    ok ".venv created ($("$ROOT/.venv/bin/python" --version))"
  fi
}

# ---------------------------------------------------------------- stage 5
stage_pydeps() {
  CURRENT_STAGE="5/9 Python packages (pinned)"; stage "$CURRENT_STAGE"
  local req="requirements/train.txt" flag=""
  [ "$MINIMAL" = "1" ] && { req="requirements/base.txt"; flag="--minimal"; }
  if [ "$CHECK_ONLY" = "1" ]; then
    [ -x "$ROOT/.venv/bin/python" ] && "$ROOT/.venv/bin/python" scripts/check_env.py $flag || warn "Python packages not (fully) installed yet"
    return 0
  fi
  info "installing from $req (torch is large; the first run can take several minutes)"
  "$ROOT/.tools/bin/uv" pip install --python "$ROOT/.venv/bin/python" -r "$req"
  # shellcheck disable=SC2086
  "$ROOT/.venv/bin/python" scripts/check_env.py $flag
  ok "Python environment verified"
}

# ---------------------------------------------------------------- stage 6
stage_javadeps() {
  CURRENT_STAGE="6/9 ONNX Runtime Java jar"; stage "$CURRENT_STAGE"
  if [ "$SKIP_JAVA_DEPS" = "1" ]; then warn "skipped (--skip-java-deps)"; return 0; fi
  if [ "$CHECK_ONLY" = "1" ]; then
    ls java/lib/onnxruntime-*.jar >/dev/null 2>&1 && ok "ONNX Runtime jar present" || warn "ONNX Runtime jar not downloaded yet"
    return 0
  fi
  bash "$ROOT/scripts/fetch_java_deps.sh"
}

# ---------------------------------------------------------------- stage 7
stage_smoke() {
  CURRENT_STAGE="7/9 ONNX integration smoke test"; stage "$CURRENT_STAGE"
  if [ "$CHECK_ONLY" = "1" ] || [ "$SKIP_SMOKE" = "1" ]; then warn "skipped"; return 0; fi
  if bash "$ROOT/scripts/smoke_onnx.sh"; then
    ok "Python -> ONNX -> Java integration works on this machine"
  else
    fail "The install finished but the ONNX smoke test FAILED. This is exactly the risk the smoke test exists to catch."
    fail "Do not continue to training. Send the full output (and logs/ newest file) to Claude/ChatGPT."
    return 1
  fi
}

# ---------------------------------------------------------------- stage 8
stage_report() {
  CURRENT_STAGE="8/9 environment report"; stage "$CURRENT_STAGE"
  [ "$CHECK_ONLY" = "1" ] && { warn "skipped"; return 0; }
  # shellcheck disable=SC1091
  . "$ROOT/env.sh"
  {
    echo "Environment report generated $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "host: $(uname -a)"
    [ "$MP_OS" = "Darwin" ] && echo "macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "git: $(git --version)"
    echo "javac: $(javac -version 2>&1)"; echo "java: $(java -version 2>&1 | head -1)"
    echo "JAVA_HOME: ${JAVA_HOME:-unset}"
    echo "python: $(python --version 2>&1) ($(command -v python))"
    echo "uv: $("$ROOT/.tools/bin/uv" --version)"
    ls java/lib/onnxruntime-*.jar >/dev/null 2>&1 && echo "ort jar sha256: $(sha256_of "$(ls java/lib/onnxruntime-*.jar | head -1)")"
    echo "--- pip freeze ---"; "$ROOT/.tools/bin/uv" pip freeze --python "$ROOT/.venv/bin/python"
  } > "$ROOT/logs/env_report.txt"
  ok "wrote logs/env_report.txt (send this file to ChatGPT/Claude when asking for help)"
}

# ---------------------------------------------------------------- stage 9
stage_git() {
  CURRENT_STAGE="9/9 git"; stage "$CURRENT_STAGE"
  [ "$CHECK_ONLY" = "1" ] && { warn "skipped"; return 0; }
  local top=""
  if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    top="$(git -C "$ROOT" rev-parse --show-toplevel)"
    if [ "$top" != "$ROOT" ]; then
      warn "This folder is inside another git repository ($top). Not creating a nested repo."; return 0
    fi
    ok "git repository already initialised"
  else
    git -C "$ROOT" init -q -b main 2>/dev/null || { git -C "$ROOT" init -q; git -C "$ROOT" symbolic-ref HEAD refs/heads/main; }
    ok "git repository initialised (branch main)"
  fi
  if ! git -C "$ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
    if [ -n "$(git -C "$ROOT" config user.name || true)" ] && [ -n "$(git -C "$ROOT" config user.email || true)" ]; then
      git -C "$ROOT" add -A
      git -C "$ROOT" commit -q -m "chore: initial scaffold (milestones 1-2: environment and ONNX smoke test)"
      ok "created initial commit"
    else
      warn "git user.name/email not configured, so no commit was made (the repository IS initialised). Run once:"
      warn "  git config --global user.name \"Your Name\"; git config --global user.email \"you@example.com\""
      warn "then:  git add -A && git commit -m \"chore: initial scaffold (milestones 1-2)\""
    fi
  fi
}

main() {
  printf 'Marine Plastic Detection and Coastal Cleanup Planner -- setup\n'
  printf 'project folder: %s\n' "$ROOT"
  [ "$CHECK_ONLY" = "1" ] && printf 'MODE: --check (nothing will be changed)\n'
  stage_preflight
  stage_layout
  stage_jdk
  stage_python
  stage_pydeps
  stage_javadeps
  stage_smoke
  stage_report
  stage_git
  stage "RESULT"
  if [ "$CHECK_ONLY" = "1" ]; then
    ok "check finished (no changes made)"
  else
    ok "SETUP COMPLETE"
    printf '\nNext:\n  1) source env.sh                  # puts JDK 21 + the Python venv on PATH in this terminal\n'
    printf '  2) bash scripts/run_tests.sh      # the single test command\n'
    printf '  3) send logs/env_report.txt and the test output back to Claude/ChatGPT\n'
  fi
}

main 2>&1 | tee -a "logs/setup-$(date +%Y%m%d-%H%M%S).log"
exit "${PIPESTATUS[0]}"
