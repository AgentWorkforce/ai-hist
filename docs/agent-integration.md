# Agent integration

RelayHistory 1.0 has one production path: the TypeScript SDK, CLI, and MCP
server call the mandatory Rust Node-API engine. Reads are cache-only; provider
I/O happens only when an integration explicitly requests shallow discovery or
full sync.

Use these public operations:

- `listSessionCatalogPage()` / MCP `list_sessions` for bounded cache-only
  session discovery results.
- `discoverSessions()` / MCP `discover_sessions` to refresh shallow provider
  metadata.
- `search()`, `recent()`, `getSession()`, and `getSessionEventsPage()` for
  indexed history reads.
- `sync()` / MCP `sync` for explicit full local ingestion.

The CLI equivalents are `sessions list`, `sessions discover`, `search`,
`recent`, `session`, `events`, `stats`, and `sync`. See
[Session catalog](session-catalog.md) for discovery and pagination contracts
and [Architecture](architecture.md) for the process boundary.

The old cloud push, login, Pair, hook installer, tag, and trajectory convenience
commands were removed in 1.0. They are not available through subprocess or
JavaScript fallbacks; see the [migration guide](native-sdk-migration.md).
