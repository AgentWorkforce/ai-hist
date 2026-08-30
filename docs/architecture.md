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
version 2 and catalog contract version 1, normalizes nullable fields, maps
native errors, and supplies pagination helpers.

The CLI and MCP server import only the SDK's public functions. They do not
open SQLite, import `ai-hist-native`, scan providers, or invoke another CLI.

## Operation semantics

| Operation | Provider I/O | Database work | Missing database |
|---|---:|---|---|
| `listSessionCatalog*` | none | one indexed cache query | empty page |
| `discoverSessions` | bounded shallow reads | catalog upserts | creates catalog DB |
| `search`, `recent`, `getSession`, `stats` | none | indexed reads | empty result |
| `getSessionEventsPage` | none | bounded keyset page | empty page |
| `sync` | full explicit scan | migrations + ingestion | creates DB |

No read operation invokes discovery or sync. A common cold start is:

```ts
await discoverSessions({ limit: 100 });
const sessions = await listSessionCatalog({ limit: 100 });
```

Events use `(ts_ms, id)` keyset pagination. Catalog ordering is
`(last_activity_ms DESC, source ASC, session_id ASC)`, with null timestamps at
the tail. These total orders prevent duplicate or omitted rows at timestamp
ties.

## Native errors

The SDK distinguishes unsupported platform, supported platform package
missing, addon load failure, native/SDK contract mismatch, database-open
failure, invalid argument, query failure, discovery failure, and sync failure.
There is no alternate runtime after any native-load error.
