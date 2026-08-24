#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export AI_HIST_DB="$TMP/ai-history.db"
export TRAJECTORY_ROOT="$TMP/trajectories"
export OPENCODE_DB="$TMP/opencode.db"
export HOME="$TMP/home"
mkdir -p "$HOME/.claude/projects/e2e-project" "$HOME/.codex/sessions/2026/06/20" "$TRAJECTORY_ROOT/planner/compacted"
mkdir -p "$HOME/.grok/sessions/%2Ftmp%2Fe2e%2Fgrok/grok-e2e"
mkdir -p "$HOME/.cursor/projects/tmp-e2e-cursor/agent-transcripts/cursor-e2e"

rust_ai_hist() {
  if [[ -n "${AI_HIST_RUST_BIN:-}" ]]; then
    "$AI_HIST_RUST_BIN" "$@"
  else
    cargo run -q -p ai-hist-cli --manifest-path "$ROOT/Cargo.toml" -- "$@"
  fi
}

cat > "$HOME/.claude/history.jsonl" <<'JSONL'
{"display":"e2e claude release tagging prompt","timestamp":1700000000000,"project":"/tmp/e2e/project","sessionId":"claude-e2e"}
JSONL

cat > "$HOME/.codex/history.jsonl" <<'JSONL'
{"text":"e2e codex release tagging prompt","ts":1700000001,"session_id":"codex-e2e"}
JSONL

cat > "$HOME/.codex/sessions/2026/06/20/rollout-codex-e2e.jsonl" <<'JSONL'
{"type":"session_meta","payload":{"id":"codex-e2e","cwd":"/tmp/e2e/codex","git":{"branch":"main"}}}
JSONL

cat > "$HOME/.grok/sessions/%2Ftmp%2Fe2e%2Fgrok/grok-e2e/summary.json" <<'JSON'
{"info":{"id":"grok-e2e","cwd":"/tmp/e2e/grok"},"created_at":"2026-06-20T10:02:00.000Z","updated_at":"2026-06-20T10:03:00.000Z","head_branch":"main"}
JSON
cat > "$HOME/.grok/sessions/%2Ftmp%2Fe2e%2Fgrok/grok-e2e/chat_history.jsonl" <<'JSONL'
{"type":"user","content":[{"type":"text","text":"e2e grok release tagging prompt"}]}
{"type":"user","synthetic_reason":"system_reminder","content":[{"type":"text","text":"do not import this synthetic prompt"}]}
{"type":"assistant","content":[{"type":"text","text":"grok assistant summary"}]}
JSONL

cat > "$HOME/.claude/projects/e2e-project/claude-e2e.jsonl" <<'JSONL'
{"sessionId":"claude-e2e","cwd":"/tmp/e2e/project","gitBranch":"main","timestamp":"2026-06-20T10:00:00.000Z"}
{"sessionId":"claude-e2e","type":"assistant","message":{"content":[{"type":"text","text":"assistant summary"}]},"timestamp":"2026-06-20T10:01:00.000Z"}
JSONL

cat > "$HOME/.cursor/projects/tmp-e2e-cursor/agent-transcripts/cursor-e2e/cursor-e2e.jsonl" <<'JSONL'
{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\ne2e cursor release tagging prompt\n</user_query>"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}
JSONL

cat > "$TRAJECTORY_ROOT/planner/compacted/trajectory-e2e.json" <<'JSON'
{
  "id": "trajectory-e2e",
  "version": 1,
  "personaId": "planner",
  "projectId": "agent-workforce",
  "task": {
    "title": "e2e trajectory release tagging task",
    "description": "Choose release test coverage."
  },
  "status": "completed",
  "startedAt": "2026-06-06T10:00:00.000Z",
  "completedAt": "2026-06-06T10:05:00.000Z",
  "decisions": [{
    "question": "What should be tested?",
    "chosen": "full Rust parity E2E",
    "reasoning": "Fallback is no longer sufficient.",
    "alternatives": ["scoped wrapper"]
  }],
  "retrospective": {
    "summary": "Parity test selected.",
    "approach": "Exercise every source.",
    "learnings": ["Installer and sync both matter."],
    "confidence": 0.9
  }
}
JSON

sqlite3 "$OPENCODE_DB" <<'SQL'
CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT, time_created INTEGER);
CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);
INSERT INTO session VALUES ('opencode-e2e', '/tmp/e2e/opencode', 1700000002000);
INSERT INTO message VALUES ('msg-e2e', 'opencode-e2e', 1700000002000, '{"role":"user"}');
INSERT INTO part VALUES ('part-e2e', 'msg-e2e', 'opencode-e2e', 1700000002000, '{"type":"text","text":"e2e opencode release tagging prompt"}');
SQL

"$ROOT/ai-hist" sync
"$ROOT/ai-hist" tag claude-e2e release-e2e --source claude
"$ROOT/ai-hist" tag opencode-e2e release-e2e --source opencode
"$ROOT/ai-hist" tag grok-e2e release-e2e --source grok
"$ROOT/ai-hist" tag cursor-e2e release-e2e --source cursor
"$ROOT/ai-hist" tag trajectory-e2e release-e2e --source trajectory
"$ROOT/ai-hist" search release --tag release-e2e --json
"$ROOT/ai-hist" search --tag release-e2e >/dev/null
"$ROOT/ai-hist" session claude-e2e --full >/dev/null
"$ROOT/ai-hist" show 1 --json >/dev/null
"$ROOT/ai-hist" context 1 >/dev/null
"$ROOT/ai-hist" pack release --json >/dev/null
"$ROOT/ai-hist" stats --json >/dev/null
"$ROOT/ai-hist" tags --sessions --json >/dev/null

sources=$(sqlite3 "$AI_HIST_DB" "SELECT group_concat(source, ',') FROM (SELECT DISTINCT source FROM history ORDER BY source)")
test "$sources" = "claude,codex,cursor,grok,opencode,trajectory"

codex_project=$(sqlite3 "$AI_HIST_DB" "SELECT project FROM history WHERE source='codex' AND session_id='codex-e2e'")
test "$codex_project" = "/tmp/e2e/codex"

claude_session=$(sqlite3 "$AI_HIST_DB" "SELECT cwd || '|' || git_branch || '|' || last_assistant_text FROM sessions WHERE source='claude' AND session_id='claude-e2e'")
test "$claude_session" = "/tmp/e2e/project|main|assistant summary"

grok_session=$(sqlite3 "$AI_HIST_DB" "SELECT cwd || '|' || git_branch || '|' || last_assistant_text FROM sessions WHERE source='grok' AND session_id='grok-e2e'")
test "$grok_session" = "/tmp/e2e/grok|main|grok assistant summary"

synthetic=$(sqlite3 "$AI_HIST_DB" "SELECT COUNT(*) FROM history WHERE source='grok' AND prompt LIKE '%synthetic prompt%'")
test "$synthetic" = "0"

rust_ai_hist --db "$AI_HIST_DB" tag codex-e2e release-e2e --source codex
rust_ai_hist --db "$AI_HIST_DB" search release --tag release-e2e --json

(cd "$ROOT/sdk-ts" && npm test)

echo "E2E verification completed with temp DB: $AI_HIST_DB"
