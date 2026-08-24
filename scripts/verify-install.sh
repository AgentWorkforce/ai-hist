#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SOURCE_BIN="$TMP/source-bin"
SOURCE_SHARE="$TMP/source-share"
mkdir -p "$SOURCE_BIN" "$SOURCE_SHARE"
touch "$SOURCE_BIN/ai-hist-python" "$SOURCE_BIN/ai-hist-rust"
touch "$SOURCE_SHARE/ai-hist-python" "$SOURCE_SHARE/ai-hist-wrapper"
AI_HIST_SOURCE_DIR="$ROOT" \
AI_HIST_BIN_DIR="$SOURCE_BIN" \
AI_HIST_INSTALL_DIR="$SOURCE_SHARE" \
AI_HIST_BUILD_PROFILE=debug \
AI_HIST_NO_AUTOSYNC=1 \
  sh "$ROOT/install.sh"

test -x "$SOURCE_BIN/ai-hist"
test -x "$SOURCE_SHARE/ai-hist-rust-bin"
test ! -e "$SOURCE_BIN/ai-hist-rust"
test ! -e "$SOURCE_BIN/ai-hist-python"
test ! -e "$SOURCE_SHARE/ai-hist-python"
"$SOURCE_BIN/ai-hist" --help | grep -q 'Usage: ai-hist'
"$SOURCE_BIN/ai-hist" --version | grep -q '^ai-hist '

TEST_HOME="$TMP/home"
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
test "$(sqlite3 "$TEST_DB" 'SELECT COUNT(*) FROM history')" = "2"
search_output=$(HOME="$TEST_HOME" AI_HIST_DB="$TEST_DB" "$SOURCE_BIN/ai-hist" search installer --json)
grep -q 'installer claude prompt' <<<"$search_output"
grep -q 'installer codex prompt' <<<"$search_output"

FAKE_TOOLS="$TMP/fake-tools"
BINARY_BIN="$TMP/binary-bin"
BINARY_SHARE="$TMP/binary-share"
mkdir -p "$FAKE_TOOLS"
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
PATH="$FAKE_TOOLS:$PATH" \
HOME="$INSTALL_HOME" \
AI_HIST_INSTALL_METHOD=binary \
AI_HIST_BINARY_URL="file://$FAKE_BINARY" \
AI_HIST_BIN_DIR="$BINARY_BIN" \
AI_HIST_INSTALL_DIR="$BINARY_SHARE" \
AI_HIST_NO_AUTOSYNC=1 \
  sh "$ROOT/install.sh"

test "$("$BINARY_BIN/ai-hist" --version)" = "ai-hist 9.9.9"
test "$("$BINARY_BIN/ai-hist" recent)" = "[]"
test ! -e "$BINARY_BIN/ai-hist-rust"
test ! -e "$BINARY_BIN/ai-hist-python"
test ! -e "$BINARY_SHARE/ai-hist-python"
test ! -e "$INSTALL_HOME/Library/LaunchAgents/com.ai-hist.sync.plist"

echo "Installer verification completed"
