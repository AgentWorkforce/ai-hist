#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The installer and installed launcher honor several environment overrides.
# Clear inherited values so this test always exercises the artifacts it creates.
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
unset AI_HIST_CLI AI_HIST_RUST_BIN AI_HIST_DB AI_HIST_PREFIX AI_HIST_BIN_DIR
unset AI_HIST_INSTALL_DIR AI_HIST_INSTALL_METHOD AI_HIST_BINARY_URL
unset AI_HIST_SOURCE_DIR AI_HIST_REF AI_HIST_RAW_REF AI_HIST_SOURCE_REF
unset AI_HIST_VERSION AI_HIST_NO_AUTOSYNC AI_HIST_BUILD_PROFILE
unset AI_HIST_REPO_SLUG AI_HIST_REPO_URL AI_HIST_WRAPPER_SOURCE_DIR
unset OPENCODE_DB TRAJECTORY_ROOT

command -v node >/dev/null || { echo "verify-install: node is required" >&2; exit 1; }

write_legacy_files() {
  bin_dir="$1"
  share_dir="$2"
  mkdir -p "$bin_dir" "$share_dir"
  cat > "$bin_dir/ai-hist" <<'SH'
#!/bin/sh
if [ "${AI_HIST_CLI:-auto}" = "python" ]; then exit 99; fi
SH
  cat > "$bin_dir/ai-hist-python" <<'SH'
#!/usr/bin/env sh
exec python3 "/legacy/ai-hist-python" "$@"
SH
  cat > "$bin_dir/ai-hist-rust" <<'SH'
#!/usr/bin/env sh
exec "${AI_HIST_RUST_BIN:-/legacy/ai-hist-rust-bin}" "$@"
SH
  cat > "$share_dir/ai-hist-python" <<'PY'
#!/usr/bin/env python3
"""
ai-hist: Sync and search AI CLI conversation history in SQLite.
Zero dependencies — Python standard library only.
"""
PY
  cat > "$share_dir/ai-hist-wrapper" <<'PY'
#!/usr/bin/env python3
"""Rust-first ai-hist dispatcher."""
PYTHON_CLI = ROOT / "ai-hist-python"
RUST_MANIFEST = ROOT / "Cargo.toml"
PY
}

assert_legacy_removed() {
  bin_dir="$1"
  share_dir="$2"
  test ! -e "$bin_dir/ai-hist-python"
  test ! -e "$bin_dir/ai-hist-rust"
  test ! -e "$share_dir/ai-hist-python"
  test ! -e "$share_dir/ai-hist-wrapper"
  ! grep -Fq 'AI_HIST_CLI:-auto' "$bin_dir/ai-hist"
}

SOURCE_BIN="$TMP/source-bin"
SOURCE_SHARE="$TMP/source-share"
SOURCE_HOME="$TMP/source-home"
mkdir -p "$SOURCE_HOME"
write_legacy_files "$SOURCE_BIN" "$SOURCE_SHARE"
source_install_output=$(
  HOME="$SOURCE_HOME" \
  AI_HIST_SOURCE_DIR="$ROOT" \
  AI_HIST_BIN_DIR="$SOURCE_BIN" \
  AI_HIST_INSTALL_DIR="$SOURCE_SHARE" \
  AI_HIST_BUILD_PROFILE=debug \
  AI_HIST_NO_AUTOSYNC=1 \
    sh "$ROOT/install.sh"
)
grep -q 'skipping auto-sync service install' <<<"$source_install_output"

test -x "$SOURCE_BIN/ai-hist"
test -x "$SOURCE_SHARE/ai-hist-rust-bin"
assert_legacy_removed "$SOURCE_BIN" "$SOURCE_SHARE"
"$SOURCE_BIN/ai-hist" --help | grep -q 'Usage: ai-hist'
"$SOURCE_BIN/ai-hist" --version | grep -q '^ai-hist '
if AI_HIST_CLI=python "$SOURCE_BIN/ai-hist" --version >"$TMP/legacy-env.out" 2>&1; then
  echo "verify-install: AI_HIST_CLI unexpectedly succeeded" >&2
  exit 1
fi
grep -q 'AI_HIST_CLI is no longer supported' "$TMP/legacy-env.out"

ln -s "$ROOT/ai-hist" "$TMP/linked-ai-hist"
"$TMP/linked-ai-hist" --version | grep -q '^ai-hist '

TEST_HOME="$TMP/history-home"
mkdir -p "$TEST_HOME/.claude" "$TEST_HOME/.codex"
cat > "$TEST_HOME/.claude/history.jsonl" <<'JSONL'
{"display":"installer claude prompt","timestamp":1700000000000,"project":"/tmp/install","sessionId":"install-claude"}
JSONL
cat > "$TEST_HOME/.codex/history.jsonl" <<'JSONL'
{"text":"installer codex prompt","ts":1700000001,"session_id":"install-codex"}
JSONL

TEST_DB="$TMP/history.db"
HOME="$TEST_HOME" AI_HIST_DB="$TEST_DB" OPENCODE_DB="$TMP/missing-opencode.db" \
TRAJECTORY_ROOT="$TMP/trajectories" "$SOURCE_BIN/ai-hist" sync >/dev/null
search_output=$(HOME="$TEST_HOME" AI_HIST_DB="$TEST_DB" "$SOURCE_BIN/ai-hist" search installer --json)
node -e '
  const fs = require("node:fs");
  const rows = JSON.parse(fs.readFileSync(0, "utf8"));
  const actual = rows.map(({source, session_id, prompt}) => [source, session_id, prompt]).sort();
  const expected = [
    ["claude", "install-claude", "installer claude prompt"],
    ["codex", "install-codex", "installer codex prompt"],
  ];
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    console.error("unexpected installer search output", rows);
    process.exit(1);
  }
' <<<"$search_output"

FAKE_TOOLS="$TMP/fake-tools"
BINARY_BIN="$TMP/binary-bin"
BINARY_SHARE="$TMP/binary-share"
mkdir -p "$FAKE_TOOLS"
write_legacy_files "$BINARY_BIN" "$BINARY_SHARE"
cat > "$FAKE_TOOLS/cargo" <<'SH'
#!/bin/sh
echo 'cargo should not run' >&2
exit 99
SH
chmod 755 "$FAKE_TOOLS/cargo"

FAKE_BINARY="$TMP/ai-hist-test-binary"
cat > "$FAKE_BINARY" <<'SH'
#!/bin/sh
if [ "${1:-}" = "--version" ]; then echo 'ai-hist 9.9.9'; exit 0; fi
if [ "${1:-}" = "recent" ]; then echo '[]'; exit 0; fi
echo "fake ai-hist: $*"
SH
chmod 755 "$FAKE_BINARY"

INSTALL_HOME="$TMP/install-home"
binary_install_output=$(
  PATH="$FAKE_TOOLS:$PATH" \
  HOME="$INSTALL_HOME" \
  AI_HIST_INSTALL_METHOD=binary \
  AI_HIST_BINARY_URL="file://$FAKE_BINARY" \
  AI_HIST_BIN_DIR="$BINARY_BIN" \
  AI_HIST_INSTALL_DIR="$BINARY_SHARE" \
  AI_HIST_NO_AUTOSYNC=1 \
    sh "$ROOT/install.sh"
)
grep -q 'skipping auto-sync service install' <<<"$binary_install_output"

test "$("$BINARY_BIN/ai-hist" --version)" = "ai-hist 9.9.9"
test "$("$BINARY_BIN/ai-hist" recent)" = "[]"
assert_legacy_removed "$BINARY_BIN" "$BINARY_SHARE"

cat > "$BINARY_BIN/ai-hist-rust" <<'SH'
#!/bin/sh
echo "custom launcher using ${AI_HIST_RUST_BIN:-custom-ai-hist}"
SH
cat > "$BINARY_BIN/ai-hist-python" <<'SH'
#!/bin/sh
exec python3 "/opt/custom/history-tool.py" "$@"
SH
preserve_output=$(
  PATH="$FAKE_TOOLS:$PATH" \
  HOME="$INSTALL_HOME" \
  AI_HIST_INSTALL_METHOD=binary \
  AI_HIST_BINARY_URL="file://$FAKE_BINARY" \
  AI_HIST_BIN_DIR="$BINARY_BIN" \
  AI_HIST_INSTALL_DIR="$BINARY_SHARE" \
  AI_HIST_NO_AUTOSYNC=1 \
    sh "$ROOT/install.sh" 2>&1
)
grep -q 'left unrecognized file untouched' <<<"$preserve_output"
grep -q 'custom launcher using' "$BINARY_BIN/ai-hist-rust"
grep -q '/opt/custom/history-tool.py' "$BINARY_BIN/ai-hist-python"

rm -f "$BINARY_BIN/ai-hist-rust"
ln -s "$TMP/missing-legacy-target" "$BINARY_BIN/ai-hist-rust"
symlink_output=$(
  PATH="$FAKE_TOOLS:$PATH" \
  HOME="$INSTALL_HOME" \
  AI_HIST_INSTALL_METHOD=binary \
  AI_HIST_BINARY_URL="file://$FAKE_BINARY" \
  AI_HIST_BIN_DIR="$BINARY_BIN" \
  AI_HIST_INSTALL_DIR="$BINARY_SHARE" \
  AI_HIST_NO_AUTOSYNC=1 \
    sh "$ROOT/install.sh" 2>&1
)
grep -q 'left symlink untouched' <<<"$symlink_output"
test -L "$BINARY_BIN/ai-hist-rust"
test "$(readlink "$BINARY_BIN/ai-hist-rust")" = "$TMP/missing-legacy-target"

echo "Installer verification completed"
