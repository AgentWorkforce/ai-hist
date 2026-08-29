# Changelog

All notable changes to `ai-hist` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `listSessionCatalog({ sources?, limit?, beforeMs? })` — the materialized
  session catalog, newest first, as one indexed query over the `sessions` table.
  No provider transcript is opened and neither `history` nor `session_events` is
  scanned, so it stays fast on first paint with thousands of sessions. Rows come
  back as `CatalogSession`: cwd, git branch, first/last activity, derived
  `firstPrompt`, observed `models`, originator, agent version, repo identity,
  `sourceStamp`, and `discoveryState` (`'shallow'` for a catalog-only row,
  `'full'` once a sync ingested the transcript; a `NULL` column predates the
  catalog and reads as `'full'`). `source = 'trajectory'` rows are excluded
  defensively — trajectories are derived records, not sessions. Ordering is
  `last_activity_ms DESC` with null timestamps last, then `source` and
  `session_id` so `beforeMs` keyset pagination is stable. A configured
  `projectScope` constrains rows by `cwd`, as `getHandoff` does, so sources with
  no working directory (relay) drop out of a scoped listing. Returns `[]` in
  JSONL fallback mode: the fallback scan builds `history` only, and just the
  native discovery engine writes the catalog.
- **In-memory catalog migration** — the catalog columns (`first_prompt`,
  `models_json`, `originator`, `agent_version`, `repo_url`, `initial_commit`,
  `workspace_roots_json`, `source_stamp`, `discovery_state`) landed after 0.5.0,
  so opening an older database file now backfills each one best-effort with
  `ALTER TABLE ... ADD COLUMN` against the in-memory copy sql.js holds — the same
  trick already used for `history.git_branch`. Without it every catalog read on
  a pre-0.6 file would fail with `no such column`; with it the missing values
  simply read as `NULL`. The file on disk is never modified. The two new indexes
  (`idx_sessions_source_last`, `idx_sessions_raw_path`) are created on the copy too.
- `discoverSessions({ sources?, limit?, onSession?, onDiagnostic?, binPath?, env? })`
  — a top-level async function (not an `AiHist` method) that drives
  `ai-hist sessions discover --json` and parses its JSONL stream into
  `{ sessions, diagnostics, summary }`, invoking `onSession` progressively as
  rows arrive. It is top-level because discovery **writes** the on-disk database
  while an `AiHist` is an in-memory snapshot: a method would return a reader that
  cannot see the rows it just wrote, so re-open before listing. The limit is
  global across providers and applied by recency, matching the native engine.
  Binary discovery reuses `resolveAiHistBinary` from the cloud-push path
  (`$AI_HIST_RUST_BIN` → the install.sh location → `ai-hist` on `PATH`). Failure
  behavior mirrors the CLI: a provider that fails contributes a diagnostic and
  the run still resolves; the promise rejects only when the binary cannot be run
  (with an error naming `AI_HIST_RUST_BIN`) or on a non-zero exit, whose stderr
  is included. Unparseable or unknown JSONL lines are skipped rather than
  failing the run, so a newer binary's extra line types are forward-compatible.
- `SESSION_CATALOG_CONTRACT_VERSION` (currently `1`), mirroring the native
  constant, plus the `CatalogSession`, `ListCatalogOptions`,
  `DiscoverSessionsOptions`, `DiscoverResult`, `DiscoverySummary`,
  `DiscoveryDiagnostic`, `DiscoveryCounters`, `ProviderDiscoverySummary`, and
  `SourceExemption` types.
- **MCP tool `list_sessions`** (read-only) — `sources?`, `limit` (default 20,
  max 200), `before_ms`; returns the catalog rows as JSON alongside
  `contractVersion`, `sourceKind`, `dbPath`, and `projectScope`. Its `sources`
  enum omits `trajectory`, which the catalog never contains.

### Changed

- `listSessions` is unchanged and still derives sessions from `history`; its doc
  comment now points at `listSessionCatalog` as the fast and complete path
  (the catalog also holds sessions that only shallow discovery has seen).

## [0.5.0] - 2026-08-20

### Added

- `getSessionEvents(sessionId, { source? })` — the normalized transcript for one
  session from the `session_events` table: user/assistant text, thinking, tool
  calls, and tool results in order, with parsed per-event token usage
  (`tokenUsage`). Returns `[]` on databases that predate the table and in JSONL
  fallback mode. On large databases prefer streaming from the native CLI
  (`ai-hist events <session-id> --json`); this SDK loads the whole file into
  memory.
- `getToolCalls(sessionId, { source? })` — the session's `tool_calls` rows with
  typed `isError`.
- `SessionEvent` and `SessionToolCall` types.

## [0.4.1] - 2026-07-07

### Added

- **In-process cloud push (`ai-hist/cloud` → `pushToCloud`)** — a thin wrapper
  that drives the real `ai-hist push --json` Rust binary and parses its result,
  rather than re-implementing the push pipeline in TypeScript. The binary stays
  the single source of truth for batching/cursor/dedup; the SDK just gives hosts
  (e.g. the Agent Relay runtime) an ergonomic surface to sync in-process without
  shelling out to a CLI by hand. Binary discovery: `$AI_HIST_RUST_BIN` → the
  install.sh location → `ai-hist` on `PATH`. Also exports `resolveAiHistBinary`.

## [0.3.7] - 2026-06-27

### Added

- **Add cloud-client module with loginCloud and loadStoredRelayhistoryAuth**

## [0.3.5] - 2026-06-24

### Added

- **Add grok history source**

### Fixed

- Address grok review feedback

### Documentation

- Pair + cloud-sync guides + Pair client SDK/hook (#25)

## [0.3.4] - 2026-06-20

### Changed

- Make the public `ai-hist` command Rust-default for the user-facing CLI
  surface, including sync, show/context/session, stats, pack, resume,
  export/import, and tagging commands.
- Add a one-command installer that builds and installs deterministic `ai-hist`,
  `ai-hist-rust`, and `ai-hist-python` launchers without requiring users to run
  Cargo commands manually.
- Keep the legacy Python CLI as an explicit compatibility escape hatch via
  `AI_HIST_CLI=python` or `ai-hist-python`.

### Fixed

- Align Rust default database path with `XDG_DATA_HOME`.
- Create legacy session metadata schema from Rust DB initialization.
- Set WAL mode from Rust DB initialization.
- Keep the legacy Python fallback importable on Python 3.9.6 by avoiding PEP
  604-only annotations.

## [0.3.2] - 2026-06-12

### Added

- **Add MCP project scope argument**

### Dependencies

- Apply pr-reviewer fixes for #11 (#11)
- Apply pr-reviewer fixes for #11 (#11)

## [0.3.1] - 2026-06-06

### Added

- **Add ai-hist-mcp wrapper package**
- **Add ai-hist MCP trajectory source**
- **Add TypeScript MCP server and Smithery config for ai-hist-mcp**

### Documentation

- Keep MCP local-first

### Dependencies

- Apply pr-reviewer fixes for #9 (#9)
- Apply pr-reviewer fixes for #9 (#9)

## [0.2.3] - 2026-05-22

### Changed

- Rewrite listSessions to use window functions (~68x faster)

## [0.2.1] - 2026-05-22

### Added

- **Native JSONL fallback — SDK works without the Python CLI**
