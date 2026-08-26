#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v sqlite3 >/dev/null || { echo "verify-e2e: sqlite3 is required" >&2; exit 1; }
command -v node >/dev/null || { echo "verify-e2e: node is required" >&2; exit 1; }
unset AI_HIST_CLI

assert_eq() {
  local label="$1" actual="$2" expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL $label: got '$actual', want '$expected'" >&2
    exit 1
  fi
}

# Print one field of a --json payload piped on stdin. The argument is a JS
# expression over `d`, the parsed document -- enough to name a single field
# without pulling a JSON query tool into the script's dependencies.
json_field() {
  node -e '
    const d = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
    process.stdout.write(String(eval(process.argv[1])));
  ' "$1"
}

export AI_HIST_DB="$TMP/ai-history.db"
export TRAJECTORY_ROOT="$TMP/trajectories"
export OPENCODE_DB="$TMP/opencode.db"
export HOME="$TMP/home"
unset RELAYCAST_API_KEY RELAYCAST_WORKSPACE_ID RELAYCAST_BASE_URL
mkdir -p "$HOME/.claude/projects/e2e-project" "$HOME/.codex/sessions/2026/06/20" "$TRAJECTORY_ROOT/planner/compacted"
mkdir -p "$HOME/.grok/sessions/%2Ftmp%2Fe2e%2Fgrok/grok-e2e"
mkdir -p "$HOME/.cursor/projects/tmp-e2e-cursor/agent-transcripts/cursor-e2e"

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

sqlite3 -bail "$OPENCODE_DB" <<'SQL'
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
session_prompt=$("$ROOT/ai-hist" session claude-e2e --full --json | json_field 'd.map((e) => e.prompt).join("|")')
assert_eq "session prompts" "$session_prompt" "e2e claude release tagging prompt"

show_resume=$("$ROOT/ai-hist" show 1 --json | json_field 'd.resume_cmd')
assert_eq "show resume command" "$show_resume" "cd /tmp/e2e/project && claude --resume claude-e2e"

# `context` has no --json; assert it marks exactly the requested entry.
context_out=$("$ROOT/ai-hist" context 1)
context_focus=$(printf '%s\n' "$context_out" | grep -c '>>>' || true)
assert_eq "context focus rows" "$context_focus" "1"

pack_sources=$("$ROOT/ai-hist" pack release --json | json_field 'd.entries.map((e) => e.source).sort().join(",")')
assert_eq "pack sources" "$pack_sources" "claude,codex,cursor,grok,opencode,trajectory"

stats_total=$("$ROOT/ai-hist" stats --json | json_field 'd.total')
assert_eq "stats total" "$stats_total" "6"

tagged_sessions=$("$ROOT/ai-hist" tags --sessions --json | json_field 'd.find((t) => t.name === "release-e2e").session_count')
assert_eq "tagged session count" "$tagged_sessions" "5"

sources=$(sqlite3 "$AI_HIST_DB" "SELECT DISTINCT source FROM history" | sort)
expected_sources=$(printf '%s\n' claude codex cursor grok opencode trajectory)
assert_eq "sources" "$sources" "$expected_sources"

codex_project=$(sqlite3 "$AI_HIST_DB" "SELECT project FROM history WHERE source='codex' AND session_id='codex-e2e'")
assert_eq "codex project" "$codex_project" "/tmp/e2e/codex"

claude_session=$(sqlite3 -separator '|' "$AI_HIST_DB" "SELECT coalesce(cwd, '<NULL>'), coalesce(git_branch, '<NULL>'), coalesce(last_assistant_text, '<NULL>') FROM sessions WHERE source='claude' AND session_id='claude-e2e'")
assert_eq "claude session metadata" "$claude_session" "/tmp/e2e/project|main|assistant summary"

grok_session=$(sqlite3 -separator '|' "$AI_HIST_DB" "SELECT coalesce(cwd, '<NULL>'), coalesce(git_branch, '<NULL>'), coalesce(last_assistant_text, '<NULL>') FROM sessions WHERE source='grok' AND session_id='grok-e2e'")
assert_eq "grok session metadata" "$grok_session" "/tmp/e2e/grok|main|grok assistant summary"

synthetic=$(sqlite3 "$AI_HIST_DB" "SELECT COUNT(*) FROM history WHERE source='grok' AND prompt LIKE '%synthetic prompt%'")
assert_eq "grok synthetic prompt count" "$synthetic" "0"

"$ROOT/ai-hist" --db "$AI_HIST_DB" tag codex-e2e release-e2e --source codex
"$ROOT/ai-hist" --db "$AI_HIST_DB" search release --tag release-e2e --json

echo "E2E verification completed with temp DB: $AI_HIST_DB"
