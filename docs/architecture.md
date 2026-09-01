# Production architecture

RelayHistory has one production call graph:

```text
provider files / SQLite
        │
        ▼
ai-hist-core + Rust ingestion engine
        │ typed Rust functions
        ▼
ai-hist-native (Node-API, async worker tasks)
        │ typed native objects
        ▼
ai-hist TypeScript SDK
        ├── ai-hist Node CLI
        └── ai-hist MCP server
```

Rust owns provider discovery/parsing, schema creation and migration, direct
SQLite connections, catalog queries, history/event queries, search,
statistics, and sync. Blocking filesystem and SQLite work is dispatched away
from Node's event loop. TypeScript validates inputs, validates native contract
version 4, catalog contract version 2, and hydration contract version 1, normalizes nullable fields, maps
native errors, and supplies pagination helpers.

The CLI and MCP server import only the SDK's public functions. They do not
open SQLite, import `ai-hist-native`, scan providers, or invoke another CLI.

## Session ledger and location scope

There is one session ledger. `local` and `remote` are presences recording where
a logical session was observed, not independent session stores. Collection
operations accept one scope: `local` (the default), `remote`, or `all`. The
`all` view is the union of both presences, deduplicated by canonical session
identity, so materializing a remote session locally does not create a second
user-visible session. Connector-specific locator, change stamp, and discovery
state live on each presence, preventing a local scan and a cloud scan from
overwriting one another's acquisition state.

Scope changes selection, not I/O. Cached collection reads (`sessions list`,
`search`, and `recent`) stay database-only for every scope. Direct session and
event lookup already names one session and remains scope-independent.

Acquisition is still explicit. Local discovery and sync scan provider files;
remote discovery and sync run provider connectors (`claude-web` for
claude.ai/code web sessions, `codex-cloud` for Codex cloud tasks — see
[Remote connectors](remote-connectors.md)). A connector participates only when
the provider CLI's stored sign-in is present on the machine. Explicit `remote`
acquisition with no connector configured returns an unsupported operation; the
engine must not silently fall back to local work. `all` acquisition runs the
local adapters plus every configured connector. An acquisition result's
`scope` records the request; its `locations_run` records which connector
locations actually executed, and observed presences belong to each session
row's `locations`.

## Operation semantics

| Operation | Provider I/O | Database work | Missing database |
|---|---:|---|---|
| `listSessionCatalog*` (`local` / `remote` / `all`) | none | one indexed cache query | empty page |
| `discoverSessions` (`local`, default) | bounded shallow reads | catalog upserts | creates catalog DB |
| `discoverSessions` (`remote`) | configured remote connectors (error when none) | catalog upserts | creates catalog DB |
| `discoverSessions` (`all`) | local adapters + configured remote connectors | catalog upserts | creates catalog DB |
| `hydrateSession` | one selected provider session and linked evidence | transactional evidence + checkpoint upsert | `SESSION_NOT_FOUND` |
| `search`, `recent` (`local` / `remote` / `all`) | none | indexed reads | empty result |
| `stats` (`local` / `remote` / `all`) | none | indexed aggregate reads | empty result |
| `getSession` | none | indexed identity read | empty result |
| `getSessionEventsPage` | none | bounded keyset page | empty page |
| `sync` (`local`, default) | full explicit scan | migrations + ingestion | creates DB |
| `sync` (`remote`) | configured remote connectors (error when none) | catalog upserts + presences | creates DB |
| `sync` (`all`) | full local scan + configured remote connectors | migrations + ingestion | creates DB |

No read operation invokes discovery or sync. A common cold start is:

```ts
await discoverSessions({ limit: 100, scope: 'local' });
const sessions = await listSessionCatalog({ limit: 100, scope: 'all' });
await hydrateSession({ source: sessions[0].source, sessionId: sessions[0].sessionId });
```

Global sync owns enumeration while targeted hydration resolves one persisted
catalog presence. Both call the same Rust provider normalization helpers.
TypeScript never parses a provider source or opens SQLite. Per-session
checkpoints make unchanged calls constant-work after source resolution, and
`session_relationships` preserves Codex child identities without flattening
their events into the selected root.

Events use `(ts_ms, id)` keyset pagination. Catalog ordering is
`(last_activity_ms DESC, source ASC, session_id ASC)`, with null timestamps at
the tail. These total orders prevent duplicate or omitted rows at timestamp
ties.

## Native errors

The SDK distinguishes unsupported platform, supported platform package
missing, addon load failure, native/SDK contract mismatch, database-open
failure, invalid argument, query failure, discovery failure, and sync failure.
There is no alternate runtime after any native-load error.
