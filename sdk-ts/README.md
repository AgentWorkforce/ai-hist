# ai-hist

The public TypeScript SDK, Node CLI, and MCP server for RelayHistory. Every
operation uses the mandatory `ai-hist-native` Node-API engine; there is no
JavaScript SQLite implementation or provider-file fallback.

```bash
npm install ai-hist
```

```ts
import {
  discoverSessions,
  listSessionCatalogPage,
  getSession,
  getSessionEventsPage,
  sessionEvents,
  search,
  recent,
  stats,
  sync,
} from 'ai-hist';

await discoverSessions({ limit: 100 });
const page = await listSessionCatalogPage({ limit: 20 });
const prompts = await getSession(page.sessions[0].sessionId);
const events = await getSessionEventsPage(page.sessions[0].sessionId, { limit: 200 });

for await (const event of sessionEvents(page.sessions[0].sessionId)) {
  consume(event);
}
```

All APIs are async. `listSessionCatalog` and `listSessionCatalogPage` are
cache-only. `discoverSessions` is shallow discovery. `sync` is full ingestion.
Missing databases return empty read results; they do not trigger provider I/O.

The event primitive is page-based and uses `{ tsMs, id }` as a deterministic
cursor. The `sessionEvents` async iterator walks pages without accumulating a
large transcript. `getSessionEvents` is an explicit collecting convenience.

Native loading failures distinguish unsupported platforms, missing optional
platform packages, addon load failures, SDK/native contract mismatches, and
database open failures through stable `RelayHistoryError` subclasses.

The old synchronous `AiHist` class and `openAiHist()` API were removed in 1.0.
See [the migration guide](../docs/native-sdk-migration.md).
