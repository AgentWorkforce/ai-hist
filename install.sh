#!/usr/bin/env sh
set -eu

REPO_SLUG="${AI_HIST_REPO_SLUG:-AgentWorkforce/relayhistory}"
REPO_URL="${AI_HIST_REPO_URL:-https://github.com/$REPO_SLUG.git}"
REF="${AI_HIST_REF:-main}"
VERSION="${AI_HIST_VERSION:-latest}"
PREFIX="${AI_HIST_PREFIX:-$HOME/.local}"
BIN_DIR="${AI_HIST_BIN_DIR:-$PREFIX/bin}"
INSTALL_DIR="${AI_HIST_INSTALL_DIR:-$PREFIX/share/ai-hist}"
BUILD_PROFILE="${AI_HIST_BUILD_PROFILE:-release}"
INSTALL_METHOD="${AI_HIST_INSTALL_METHOD:-auto}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ai-hist installer: missing required command: $1" >&2
    return 1
  fi
}

info() {
  echo "ai-hist installer: $*"
}

warn() {
  echo "ai-hist installer: $*" >&2
}

script_dir() {
  case "$0" in
    */*) cd "$(dirname "$0")" && pwd ;;
    *) pwd ;;
  esac
}

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT INT TERM

case "$INSTALL_METHOD" in
  auto | binary | source) ;;
  *)
    echo "ai-hist installer: AI_HIST_INSTALL_METHOD must be auto, binary, or source" >&2
    exit 2
    ;;
esac

platform_asset() {
  os="$(uname -s 2>/dev/null || echo unknown)"
  arch="$(uname -m 2>/dev/null || echo unknown)"

  case "$os" in
    Darwin) os_part="darwin" ;;
    Linux) os_part="linux" ;;
    *)
      warn "unsupported OS for prebuilt binary: $os"
      return 1
      ;;
  esac

  case "$arch" in
    arm64 | aarch64) arch_part="arm64" ;;
    x86_64 | amd64) arch_part="x64" ;;
    *)
      warn "unsupported architecture for prebuilt binary: $arch"
      return 1
      ;;
  esac

  echo "ai-hist-$os_part-$arch_part"
}

release_download_url() {
  asset="$1"
  if [ -n "${AI_HIST_BINARY_URL:-}" ]; then
    echo "$AI_HIST_BINARY_URL"
    return 0
  fi

  if [ "$VERSION" = "latest" ]; then
    echo "https://github.com/$REPO_SLUG/releases/latest/download/$asset"
    return 0
  fi

  echo "https://github.com/$REPO_SLUG/releases/download/$(version_tag)/$asset"
}

version_tag() {
  if [ "$VERSION" != "latest" ]; then
    case "$VERSION" in
      sdk-ts-v*) echo "$VERSION" ;;
      v*) echo "sdk-ts-$VERSION" ;;
      *) echo "sdk-ts-v$VERSION" ;;
    esac
    return 0
  fi

  echo "$REF"
}

raw_ref() {
  if [ -n "${AI_HIST_RAW_REF:-}" ]; then
    echo "$AI_HIST_RAW_REF"
    return 0
  fi

  version_tag
}

source_ref() {
  if [ -n "${AI_HIST_SOURCE_REF:-}" ]; then
    echo "$AI_HIST_SOURCE_REF"
    return 0
  fi

  raw_ref
}

install_binary_launchers() {
  rust_bin="$1"

  mkdir -p "$BIN_DIR" "$INSTALL_DIR"

  cat > "$BIN_DIR/ai-hist" <<EOF
#!/usr/bin/env sh
if [ -n "\${AI_HIST_CLI:-}" ]; then
  echo "ai-hist: AI_HIST_CLI is no longer supported; the Python CLI was removed." >&2
  exit 2
fi
exec "\${AI_HIST_RUST_BIN:-$rust_bin}" "\$@"
EOF

  chmod 755 "$BIN_DIR/ai-hist"
}

remove_if_legacy() {
  path="$1"
  max_bytes="$2"
  shift 2
  if [ ! -f "$path" ] || [ -L "$path" ]; then
    return 0
  fi
  size=$(wc -c < "$path" | tr -d ' ')
  if [ "$size" -gt "$max_bytes" ]; then
    warn "left unrecognized file untouched: $path"
    return 0
  fi
  for marker in "$@"; do
    if grep -Fq "$marker" "$path" 2>/dev/null; then
      rm -f "$path"
      info "removed legacy launcher: $path"
      return 0
    fi
  done
  warn "left unrecognized file untouched: $path"
}

remove_legacy_launchers() {
  remove_if_legacy "$BIN_DIR/ai-hist-python" 4096 "exec python3" "ai-hist-python was not installed"
  remove_if_legacy "$BIN_DIR/ai-hist-rust" 4096 "AI_HIST_RUST_BIN" "ai-hist-rust-bin"
  remove_if_legacy "$INSTALL_DIR/ai-hist-python" 262144 "Zero dependencies — Python standard library only."
  remove_if_legacy "$INSTALL_DIR/ai-hist-wrapper" 32768 "Rust-first ai-hist dispatcher."

  default_bin="$HOME/.local/bin"
  if [ "$BIN_DIR" != "$default_bin" ] && \
     { [ -e "$default_bin/ai-hist-python" ] || [ -e "$default_bin/ai-hist-rust" ]; }; then
    warn "legacy launchers remain under $default_bin; inspect and remove them manually"
  fi
}

install_prebuilt() {
  need curl || return 1

  if [ -n "${AI_HIST_BINARY_URL:-}" ]; then
    asset="ai-hist-custom"
    url="$AI_HIST_BINARY_URL"
  else
    asset="$(platform_asset)" || return 1
    url="$(release_download_url "$asset")"
  fi
  rust_bin="$INSTALL_DIR/ai-hist-rust-bin"
  download="$tmp_dir/$asset"

  info "downloading prebuilt $asset"
  mkdir -p "$INSTALL_DIR"
  if ! curl -fsSL "$url" -o "$download"; then
    warn "prebuilt binary not available at $url"
    return 1
  fi

  cp "$download" "$rust_bin"
  chmod 755 "$rust_bin"

  if ! "$rust_bin" --version >/dev/null 2>&1; then
    warn "downloaded binary failed verification"
    rm -f "$rust_bin"
    return 1
  fi

  install_binary_launchers "$rust_bin"
  info "installed prebuilt binary"
  return 0
}

resolve_source_dir() {
  if [ -n "${AI_HIST_SOURCE_DIR:-}" ]; then
    echo "$AI_HIST_SOURCE_DIR"
    return 0
  fi

  if [ -f "$(script_dir)/Cargo.toml" ] && [ -f "$(script_dir)/crates/ai-hist/Cargo.toml" ]; then
    echo "$(script_dir)"
    return 0
  fi

  need git || {
    echo "Install git or run from a cloned ai-hist checkout." >&2
    return 1
  }
  src_dir="$tmp_dir/ai-hist"
  git clone --depth 1 --branch "$(source_ref)" "$REPO_URL" "$src_dir"
  echo "$src_dir"
}

install_from_source() {
  src_dir="$(resolve_source_dir)" || exit 1

  need cargo || {
    echo "Install Rust from https://rustup.rs/ and rerun this script, or use AI_HIST_INSTALL_METHOD=binary with a published prebuilt binary." >&2
    exit 1
  }
  if [ "$BUILD_PROFILE" = "release" ]; then
    (cd "$src_dir" && cargo build --release -q -p ai-hist-cli)
  else
    (cd "$src_dir" && cargo build -q -p ai-hist-cli)
  fi

  rust_bin="$src_dir/target/$BUILD_PROFILE/ai-hist"
  if [ ! -x "$rust_bin" ]; then
    echo "ai-hist installer: Rust binary was not built at $rust_bin" >&2
    exit 1
  fi

  mkdir -p "$BIN_DIR" "$INSTALL_DIR"
  cp "$rust_bin" "$INSTALL_DIR/ai-hist-rust-bin"
  chmod 755 "$INSTALL_DIR/ai-hist-rust-bin"
  install_binary_launchers "$INSTALL_DIR/ai-hist-rust-bin"
  info "installed from source"
}

if [ "$INSTALL_METHOD" = "binary" ]; then
  install_prebuilt || exit 1
elif [ "$INSTALL_METHOD" = "source" ]; then
  install_from_source
elif [ -z "${AI_HIST_SOURCE_DIR:-}" ] && ! { [ -f "$(script_dir)/Cargo.toml" ] && [ -f "$(script_dir)/crates/ai-hist/Cargo.toml" ]; }; then
  install_prebuilt || install_from_source
else
  install_from_source
fi

remove_legacy_launchers

# Register the background sync service so history stays fresh automatically.
# Opt out with AI_HIST_NO_AUTOSYNC=1 (e.g. CI, or to manage scheduling yourself).
if [ "${AI_HIST_NO_AUTOSYNC:-0}" = "1" ]; then
  info "skipping auto-sync service install (AI_HIST_NO_AUTOSYNC=1)"
elif "$BIN_DIR/ai-hist" sync --install-service; then
  info "background sync service installed; history will sync automatically"
else
  warn "could not install the background sync service automatically"
  warn "run '$BIN_DIR/ai-hist sync --install-service' yourself, or see the README"
fi

cat <<EOF
ai-hist installed.

Commands:
  $BIN_DIR/ai-hist

Add this to your shell profile if needed:
  export PATH="$BIN_DIR:\$PATH"
EOF
