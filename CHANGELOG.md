# Changelog

Notable changes to the native `ai-hist` CLI are documented here.

## [Unreleased]

### Added

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
  `idx_sessions_source_last` and `idx_sessions_raw_path` indexes. Existing
  databases migrate in place on the next open.
- Expose `listSessions` and `discoverSessions` from the napi binding, so a Node
  host can drive the catalog in-process instead of shelling out.
- `ai-hist --version` now notices when a newer GitHub release exists and says
  so on stderr, with the install one-liner to update. The check is
  interactive-only (stderr must be a terminal), bounded by a 3-second timeout,
  and silent on any failure; suppress it with `--no-warning` or
  `AI_HIST_NO_UPDATE_CHECK=1`. Release workflows now stamp the release version
  into the binaries (`AI_HIST_RELEASE_VERSION`), so `--version` reports the
  `sdk-ts-v*` release version instead of the internal crate version.

### Breaking

- Remove the legacy Python CLI and the public `ai-hist-python` and
  `ai-hist-rust` compatibility launchers. `AI_HIST_CLI` is no longer supported;
  the source-checkout launcher exits with an explanatory error when it is set.
- Make installation Rust-only. Upgrades remove recognized installer-managed
  legacy launchers and report both removals and unrecognized files left intact.

### Changed

- Replace Python-based installer and end-to-end verification with shell,
  SQLite, Node.js, and the public Rust CLI interfaces.
