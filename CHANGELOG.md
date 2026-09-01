# Changelog

Notable changes to the native `ai-hist` CLI are documented here.

## [Unreleased]

### Breaking

- Retire standalone Rust CLI release assets and the curl/source installer.
  npm now distributes the public TypeScript SDK, Node CLI, MCP server, and
  mandatory Node-API engine.
- Rust engine consumers must recompile for the scoped session API. Public
  catalog/discovery option, page, summary, and row structs now carry scope or
  location data; that scoped-session change advanced the native/catalog
  contract versions to 3 and 2 at the time.
  The legacy-named `list_sessions_local*` and `discover_sessions_local*`
  wrappers reject non-local options instead of silently rewriting them; use
  their `*_scoped*` counterparts for `remote` or `all`.
- Human-readable history and catalog rows now include observed locations, statistics print
  the selected scope, and discovery summaries distinguish the requested scope
  from the connector locations that ran. A remote-only resume match no longer
  prints a local command; JSON reports it as unavailable and readable mode
  exits with an explanation.

### Added

- Add remote provider connectors behind the existing `--remote` / `--all`
  acquisition scopes: `claude-web` lists claude.ai/code web sessions with the
  OAuth sign-in the Claude Code CLI stored (`~/.claude/.credentials.json`,
  overridable with `RELAYHISTORY_CLAUDE_CREDENTIALS`; the endpoint moves only
  via the connector-specific `RELAYHISTORY_CLAUDE_API_BASE_URL`, guarded to
  https-or-loopback, never via the generic `ANTHROPIC_BASE_URL`), and
  `codex-cloud` lists Codex cloud tasks through `codex cloud list --json`,
  paging with `--cursor` inside the CLI's 1–20 `--limit` window
  (`~/.codex/auth.json` marks it configured). Connector rows land in the
  shared ledger as shallow catalog rows with a `remote` presence, participate
  in stamp-guarded rescans, and dedupe against local presences of the same
  session. `sessions discover --remote`, `sync --remote`, and the remote half
  of `--all` now execute configured connectors; a remote-only request on a
  machine with no connector configured keeps failing with the established
  `no remote provider connectors are configured` error, now naming each
  connector's reason. Discovery summaries gain `locations_run`, the connector
  locations that actually executed (the native-addon contract is now 4), and
  the human summary line reports it in place of the hardcoded `local`. See
  `docs/remote-connectors.md`.
- Add transactional targeted session hydration through Rust, N-API, the typed
  `hydrateSession()` SDK API, `ai-hist sessions hydrate`, and MCP
  `hydrate_session`. The result reports indexed-through state, evidence counts,
  related sessions, and bounded-work diagnostics without returning a transcript.
- Add automatic `session_hydration_checkpoints` and `session_relationships`
  migrations. Existing databases upgrade in place on their next writable open.
- Add bounded live OpenCode hydration queries keyed by session ID; targeted
  hydration never copies or scans the complete OpenCode database.
- Add a real-catalog hydration benchmark that selects provider-diverse local
  sessions and reports first-call plus unchanged-checkpoint latency.

- Add a consistent session location scope to collection operations: `--local`,
  `--remote`, and `--all` are mutually exclusive, with local as the default.
  Listing, search, recent history, statistics, packs, and resume selection
  filter one cached session ledger; `all`
  deduplicates sessions that have both local and remote presences. Direct
  session/event lookup remains scope-independent. Remote discovery and sync
  are reserved and report unsupported until provider connectors ship; `all`
  acquisition runs all configured adapters, which currently means local.
  Discovery summary `scope` is the requested acquisition scope, while each
  history/catalog row's `locations` contains observed presences; it is not a claim that a
  connector ran at every requested location.
- Add native search, recent, session, paged events, statistics, discovery,
  catalog listing, and explicit sync operations. The native-addon contract is
  now version 4.
- Add deterministic bounded event pagination using `(ts_ms, id)`.

- Add `ai-hist sessions list` and `ai-hist sessions discover`: a shallow session
  catalog over every provider. `discover` enumerates candidates cheaply, orders
  them globally by recency, and reads only bounded head/tail slices of the
  winners; `list` serves the cached catalog with one indexed query and no
  provider I/O. Both emit a versioned contract (`contract_version: 2`) —
  `list --json` as one object, `discover --json` as JSONL rows, diagnostics, and
  a closing summary with per-provider counts and operation counters. See
  `docs/session-catalog.md`.
- Extend the `sessions` catalog table with `first_prompt`, `models_json`,
  `originator`, `agent_version`, `repo_url`, `initial_commit`,
  `workspace_roots_json`, `source_stamp`, and `discovery_state`, plus the
  `idx_sessions_source_last` and `idx_sessions_raw_path` indexes. Existing
  databases migrate in place on the next open.
- Add `session_presences(source, session_id, location, raw_locator,
  source_stamp, discovery_state)`, backfill existing local evidence, and expose
  each catalog row's aggregated `locations` in catalog contract version 2.
- Expose `listSessions` and `discoverSessions` from the napi binding, so a Node
  host can drive the catalog in-process instead of shelling out.
- The npm-installed `ai-hist --version` reports the SDK package version and can
  notify interactive users when a newer npm release exists. The best-effort
  check has a 3-second timeout and is suppressed with `--no-warning` or
  `RELAYHISTORY_NO_UPDATE_CHECK=1`.

### Breaking

- Remove the legacy Python CLI and the public `ai-hist-python` and
  `ai-hist-rust` compatibility launchers. `AI_HIST_CLI` is no longer supported;
  the source-checkout launcher exits with an explanatory error when it is set.
- Make installation Rust-only. Upgrades remove recognized installer-managed
  legacy launchers and report both removals and unrecognized files left intact.

### Changed

- Recognize current Codex Desktop `response_item/message` user turns in both
  bounded session discovery and full ingestion. Existing Codex rollout indexes
  are repaired automatically, while adjacent legacy/current mirror records are
  collapsed without removing intentionally repeated prompts.
- Replace Python-based installer and end-to-end verification with shell,
  SQLite, Node.js, and the public Rust CLI interfaces.
- The opencode adapter holds its snapshot open on one connection for the whole
  run and indexes the private copy by `session_id` when the live store isn't,
  so the per-session excerpt and model queries seek instead of scanning
  `message` and `part` once per candidate. Cold shallow discovery of 1,000
  opencode sessions into a fresh database runs in ~43 ms in the native
  benchmark (was ~290 ms).
- Shallow discovery's per-candidate catalog statements (candidate
  classification, skip markers, the discovery upsert) execute through the
  prepared-statement cache, and the upsert hands back the merged catalog row
  via `RETURNING` instead of a second lookup. Discovery's catalog
  transactions commit at WAL's NORMAL durability, scoped to each transaction
  and restored before rows are emitted: discovery writes only catalog rows a
  provider rescan reproduces, while user-created records (tags, commit
  links) — including any an `on_row` callback writes through the same
  connection — keep the database's default FULL durability.
- `init_db` applies the schema in one transaction when the database needs it,
  and takes no write lock at all when the schema is already current. The
  unused `idx_sessions_cwd`, `idx_sessions_branch`, `idx_sessions_last`, and
  `idx_sessions_source_last` indexes are dropped — nothing queries them, and
  each was one more btree per catalog write.
