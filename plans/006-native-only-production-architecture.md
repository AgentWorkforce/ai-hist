# Native-only production architecture

## Current boundary map

- `ai-hist-core` already owns the schema, migrations, SQLite connection policy,
  history search/recent/session queries, event/tool/edit queries, tags, stats,
  and the small provider parsers.
- `ai-hist-cli` still owns shallow discovery, full provider ingestion/sync, and
  orchestration. Those reusable pieces must become library operations; the
  N-API layer must not contain SQL or provider logic.
- `ai-hist-napi` currently reaches through the CLI library and exposes only
  sync, cloud push, discovery, and catalog listing.
- `ai-hist` (TypeScript) currently duplicates SQLite queries with `sql.js`,
  parses provider JSON/JSONL, snapshots WAL databases through `sqlite3`, and
  shells out to the Rust CLI for discovery/cloud/pair operations.
- The MCP server consumes the `AiHist` class but also inherits those fallback
  paths. There is no public Node `ai-hist` CLI in the SDK package today.

## Implementation sequence

1. Add reusable Rust engine APIs for database selection/opening, cache-only
   catalog listing, shallow discovery, full sync, search, recent, stats,
   session history, and bounded event pagination. Move discovery types/logic
   out of the CLI-owned module (or re-export a temporary internal ingestion
   module only where a full physical move is unsafe in this release).
2. Expand `ai-hist-napi` into a typed async contract for
   `discoverSessions`, `listSessionCatalog`, `listSessionCatalogPage`,
   `getSession`, `getSessionEventsPage`, `search`, `recent`, `stats`, and
   `sync`. Every blocking call uses `spawn_blocking`; every operation accepts
   an explicit optional database path; errors carry stable codes and the
   binding exposes a native contract version.
3. Replace the TypeScript implementation with async native-backed functions,
   contract validation, platform-aware loading errors, stable SDK errors, and
   async pagination helpers. Remove `sql.js`, JSONL/trajectory scanners,
   database-buffer loading, subprocess/binary resolution, runtime download,
   and functional fallback modes.
4. Add a Node CLI to `ai-hist` whose commands only import the public SDK.
   Refactor MCP tools to call those same exported async functions; no MCP code
   may import the native package or open the database.
5. Make `ai-hist-native` an exact production dependency, add the seven tested
   platform packages (including Windows x64 MSVC), and unify package versions,
   Node-API/Node/platform documentation, build/test matrices, artifact order,
   and clean-install smoke tests. Retire the curl/standalone-binary public
   installer and release workflow.
6. Add architecture guards, cache/discovery/sync separation tests,
   deterministic catalog/event pagination tests, sparse >2 GiB on-disk tests,
   native load/contract error tests, installed CLI/MCP tests, and a documented
   benchmark command for catalog, discovery, events, CLI, and MCP.

## Deliberate compatibility breaks

- `openAiHist()` and the synchronous, snapshot-backed `AiHist` methods are
  replaced by top-level async SDK functions. No `fallback: "jsonl"` option or
  `sourceKind` exists.
- Session event retrieval becomes page-based; collecting every event is an
  explicit helper rather than the primitive operation.
- The SDK no longer honors `AI_HIST_RUST_BIN`, locates a standalone executable,
  invokes `sqlite3`, or repairs old schemas in an in-memory JavaScript copy.
  Rust migrations run when a writable database operation requires them.
- npm is the supported installation route. The public Rust CLI/curl installer
  is retired; repository-only Rust commands may remain for development while
  migration tests are completed.

## Platform gap

Windows x64 MSVC is added to the NAPI-RS matrix. Windows arm64 is not claimed
until it has a native CI runner or a reliable build-and-execute test; it remains
explicitly unsupported follow-up work.
