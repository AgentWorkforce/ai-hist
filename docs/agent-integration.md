# Agent integration

RelayHistory 1.0 has one production path: the TypeScript SDK, CLI, and MCP
server call the mandatory Rust Node-API engine. Reads are cache-only; provider
I/O happens only when an integration explicitly requests shallow discovery or
full sync.

All collection operations use the same session ledger and a shared scope enum:
`local`, `remote`, or `all`. `local` is the default. Local and remote are
presences of one logical session; `all` is a deduplicated union, not a second
query followed by concatenation.

Use these public operations:

- `listSessionCatalogPage()` / MCP `list_sessions` for bounded cache-only
  session discovery results.
- `discoverSessions()` / MCP `discover_sessions` to refresh shallow provider
  metadata.
- `search()`, `recent()`, `getSession()`, and `getSessionEventsPage()` for
  indexed history reads.
- `sync()` / MCP `sync` for explicit full local ingestion.

Pass `scope` to collection operations when the default local view is not
enough. The CLI spelling is the mutually exclusive `--local`, `--remote`, and
`--all` flags. `sessions list`, `search`, `recent`, and `stats` always query the
cached ledger and never contact a provider. Direct `getSession()` and event-page
lookups are identity-based and scope-independent.

Remote discovery and remote sync run through provider connectors
(claude.ai/code web sessions and Codex cloud tasks — see
[Remote connectors](remote-connectors.md)); with no connector configured on
the machine they fail explicitly, and integrations must surface that error
rather than retrying locally. `all` runs local adapters plus every configured
connector. Acquisition result `scope` echoes the request and `locationsRun`
reports which connector locations executed; use session `locations` for
observed presences.

The CLI equivalents are `sessions list`, `sessions discover`, `search`,
`recent`, `session`, `events`, `stats`, and `sync`. See
[Session catalog](session-catalog.md) for discovery and pagination contracts
and [Architecture](architecture.md) for the process boundary.

The old cloud push, login, Pair, hook installer, tag, and trajectory convenience
commands were removed in 1.0. They are not available through subprocess or
JavaScript fallbacks; see the [migration guide](native-sdk-migration.md).
