# Remote connectors

Remote acquisition (`sessions discover --remote`, `sync --remote`, and the
remote half of `--all`) runs through provider connectors. A connector
enumerates the sessions a provider keeps on its own service and lands them in
the shared session ledger as catalog rows with a `remote` presence. There are
two connectors:

| Connector | Source | Lists | Interface |
|---|---|---|---|
| `claude-web` | `claude` | Claude Code sessions on claude.ai/code | `GET {base}/v1/code/sessions` with the CLI's stored OAuth token |
| `codex-cloud` | `codex` | Codex cloud tasks | `codex cloud list --json` (the Codex CLI's scripting contract) |

Local and remote stay presences of one session ledger: a session observed
both by a local file adapter and by a connector is one catalog row whose
`locations` are `["local", "remote"]`.

## Configuration is the provider CLI's sign-in

RelayHistory never runs an auth flow of its own. A connector is **configured**
when the provider's own CLI has been signed in on this machine:

- `claude-web` — `~/.claude/.credentials.json` exists (the claude.ai OAuth
  token the Claude Code CLI stores at sign-in). If your setup keeps
  credentials elsewhere — for example macOS keychain-backed installs — export
  a credentials JSON file and point `RELAYHISTORY_CLAUDE_CREDENTIALS` at it.
- `codex-cloud` — `~/.codex/auth.json` exists (written by `codex login`).
  The `codex` binary must be on `PATH` when the connector runs; the CLI
  handles token refresh itself.

Requesting `--remote` acquisition with no connector configured fails loudly
with the same `no remote provider connectors are configured` error as before
connectors shipped, now including the per-connector reason. `--all` runs
whatever is configured and never errors on absence — the acquisition
summary's `locations_run` says what executed.

`claude-web` talks to `https://api.anthropic.com` only. It deliberately does
**not** honor `ANTHROPIC_BASE_URL` — that variable redirects generic
Anthropic API traffic (LLM gateways, dev proxies), and following it here
would send the stored claude.ai OAuth token to whatever host it names.
Redirecting the session-list endpoint is its own explicit decision via
`RELAYHISTORY_CLAUDE_API_BASE_URL`, and plain `http://` is refused except for
loopback addresses, so the token cannot travel cleartext to a remote host. An
expired token is reported before any request is made; running the Claude Code
CLI once refreshes it.

`codex-cloud` pages through the CLI's listing window: `codex cloud list`
accepts `--limit` values of 1–20, so the connector requests bounded pages and
follows the returned `cursor` until the listing (or a requested row cap) is
exhausted. Both connectors bound one enumeration to 100 pages (10,000 claude
sessions, 2,000 codex tasks) — bounded work is part of the discovery
contract; a later run continues from fresher listings.

## What remote rows carry

Remote listings carry less than a local transcript, and nothing is invented
to fill the gap:

- `session_id` — the provider's identifier (`session_…`/`cse_…` for claude,
  the cloud task id for codex). A Codex cloud task that is later applied or
  resumed locally produces a local rollout with its *own* session id; the two
  are separate catalog rows today.
- `first_prompt` — the provider's session/task **title**, the only
  human-readable identifier the listings offer. Both providers derive it from
  the opening prompt.
- `raw_path` — the session/task URL (`https://claude.ai/code/<id>`, the
  ChatGPT task URL).
- `repo_url` — for claude-web, the session's `git_repository` source when the
  service records one.
- Timestamps — `created_at`/`last_event_at` (claude), `updated_at` (codex).

Remote rows stay `discovery_state: "shallow"`. Neither provider serves full
transcripts through a supported listing interface, so per-message events,
tool calls, and full-text search over remote-only sessions require the
session's evidence to reach a local provider first (for example teleporting a
claude.ai session into a terminal, which writes a normal local transcript).

Claude Remote Control **bridge** sessions (`environment_kind: "bridge"`) are
views of sessions running in a local terminal; their evidence is local, so
the connector deliberately skips them rather than recording a false remote
presence.

## Change detection

Connectors participate in the same stamp-guarded rescan contract as local
adapters, per presence: the claude stamp tracks `last_event_at`, the codex
stamp tracks `updated_at` plus task `status` (an applied or failed task whose
timestamp did not move is still re-read). An unchanged remote session is
served from the catalog with zero fresh work beyond the listing itself.

## Contract stability

`codex-cloud` speaks the Codex CLI's documented scripting interface, which is
as stable as Codex chooses to keep it. `claude-web` speaks the same
session-list endpoint the Claude Code CLI's `--teleport` picker uses; that
endpoint is **not a documented public API** and can change with any Claude
Code release. A change surfaces as a connector diagnostic (or a failed
remote-only run) — never as silently missing data mixed into local results.
