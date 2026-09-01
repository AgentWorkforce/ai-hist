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

- The native-addon contract advances to 5, and Claude subagent transcripts
  whose records carry an `agentId` are now indexed under that child id instead
  of the parent's. Hydration parser version 2 re-parses and heals existing
  databases in place on the next `sessions hydrate`, moving those events from
  the parent to the child rather than duplicating them. The
  `session_relationships_v2` schema marker is required, so the first read of an
  existing database is routed through a writable open that migrates it.

### Added

- Add first-class delegation topology. `session_relationships` gains an
  identity status (`observed` or `unlinked`), child agent type, name, model and
  spawn depth, the provider evidence that established the link (kind, file
  locator, and native reference such as a Claude `toolUseId` or a Codex
  `parent_thread_id`), the provider's spawn time, and whether the child's
  events are independently addressable. Read it with the new
  `getSessionRelationships`, `getSessionTree`, and `getSessionChildrenPage`
  operations (session-relationship contract version 1), the
  `ai-hist sessions relationships` and `ai-hist sessions tree` commands, or the
  `get_session_relationships` and `get_session_tree` MCP tools. Traversal is
  pre-order, deterministically ordered by `(spawned_at_ms, relationship_uid)`,
  cycle-safe, and bounded by `max_depth` / `max_nodes`; a tree always contains
  its root, and a repeated session reached along a second path is reported as a
  cycle only when the edge points back into its own ancestry. Global `sync` now
  records Codex delegation too — including a backfill for rollouts an earlier
  version already ingested — so topology is queryable without targeted
  hydration, and existing databases migrate automatically through the
  `session_relationships_v2` marker. A full `sync` also treats a Claude
  subagent sidecar as delegated evidence rather than a session: it keeps the
  child's output under the child, and leaves the parent's own provider locator
  alone.
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
  run through the provider connectors introduced above and fail explicitly on
  a machine where none is configured; `all` acquisition runs local adapters
  plus every configured connector. Discovery summary `scope` is the requested
  acquisition scope and `locations_run` names the connector locations that
  executed, while each history/catalog row's `locations` contains observed
  presences.
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
