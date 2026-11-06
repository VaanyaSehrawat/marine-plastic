# env.sh -- activate the project toolchain in the CURRENT shell:   source env.sh
# Safe to source repeatedly (no duplicate PATH entries). Does not touch ~/.zshrc or system Java/Python.
# Works from bash and zsh.
if [ -n "${ZSH_VERSION:-}" ]; then
  eval '_mp_src="${(%):-%x}"'
else
  _mp_src="${BASH_SOURCE[0]:-$0}"
fi
MP_ROOT="$(cd "$(dirname "$_mp_src")" && pwd)"
export MP_ROOT
unset _mp_src

_mp_os="$(uname -s)"
if [ "$_mp_os" = "Darwin" ]; then
  _mp_java_home="$MP_ROOT/.tools/jdk-21/Contents/Home"
else
  _mp_java_home="$MP_ROOT/.tools/jdk-21"
fi
if [ -x "$_mp_java_home/bin/java" ]; then
  export JAVA_HOME="$_mp_java_home"
fi
export UV_PYTHON_INSTALL_DIR="$MP_ROOT/.tools/python"
export UV_PYTHON_BIN_DIR="$MP_ROOT/.tools/pybin"   # keep uv from writing shims into ~/.local/bin
export UV_CACHE_DIR="$MP_ROOT/.tools/uv-cache"
export UV_PYTHON_PREFERENCE="only-managed"
export PYTHONPATH="$MP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

_mp_prepend() {
  [ -d "$1" ] || return 0
  case ":$PATH:" in
    *":$1:"*) ;;
    *) PATH="$1:$PATH" ;;
  esac
}
_mp_prepend "$MP_ROOT/.tools/bin"
[ -n "${JAVA_HOME:-}" ] && _mp_prepend "$JAVA_HOME/bin"
if [ -d "$MP_ROOT/.venv/bin" ]; then
  export VIRTUAL_ENV="$MP_ROOT/.venv"
  _mp_prepend "$MP_ROOT/.venv/bin"
fi
export PATH
unset -f _mp_prepend
unset _mp_os _mp_java_home
