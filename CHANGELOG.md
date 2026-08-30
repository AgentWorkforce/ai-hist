# Changelog

Notable changes to the native `ai-hist` CLI are documented here.

## [Unreleased]

### Changed

- Cold shallow discovery of a large archive into a fresh database is ~3x
  faster: hot-path catalog statements run through the prepared-statement
  cache, the discovery upsert returns the merged row via `RETURNING` instead
  of a second lookup, filesystem reads run one window ahead of catalog writes
  on a persistent worker pool, and a fresh database's schema lands in a single
  transaction. The unused `idx_sessions_cwd`, `idx_sessions_branch`,
  `idx_sessions_last`, and `idx_sessions_source_last` indexes are dropped —
  nothing queries them, and each was one more btree per catalog write.
  Discovery runs commit at WAL's NORMAL durability, scoped to the run itself:
  discovery writes only catalog rows a provider rescan reproduces, while
  user-created records (tags, commit links) keep the database's default FULL
  durability.

### Breaking

- Retire standalone Rust CLI release assets and the curl/source installer.
  npm now distributes the public TypeScript SDK, Node CLI, MCP server, and
  mandatory Node-API engine.

### Added

- Add native search, recent, session, paged events, statistics, discovery,
  catalog listing, and explicit sync operations with contract version 2.
- Add deterministic bounded event pagination using `(ts_ms, id)`.

- Add `ai-hist sessions list` and `ai-hist sessions discover`: a shallow session
  catalog over every provider. `discover` enumerates candidates cheaply, orders
  them globally by recency, and reads only bounded head/tail slices of the
  winners; `list` serves the cached catalog with one indexed query and no
  provider I/O. Both emit a versioned contract (`contract_version: 1`) —
  `list --json` as one object, `discover --json` as JSONL rows, diagnostics, and
  a closing summary with per-provider counts and operation counters. See
  `docs/session-catalog.md`.
- Extend the `sessions` catalog table with `first_prompt`, `models_json`,
  `originator`, `agent_version`, `repo_url`, `initial_commit`,
  `workspace_roots_json`, `source_stamp`, and `discovery_state`, plus the
  `idx_sessions_raw_path`, `idx_sessions_recency`, and
  `idx_sessions_source_recency` indexes. Existing databases migrate in place
  on the next open.
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

- Replace Python-based installer and end-to-end verification with shell,
  SQLite, Node.js, and the public Rust CLI interfaces.
