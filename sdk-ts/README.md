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
  hydrateSession,
  listSessionCatalogPage,
  getSession,
  getSessionEventsPage,
  sessionEvents,
  getSessionToolCallsPage,
  sessionToolCalls,
  getSessionFileEditsPage,
  sessionFileEdits,
  search,
  recent,
  stats,
  sync,
} from 'ai-hist';

await discoverSessions({ scope: 'all', limit: 100 });
const page = await listSessionCatalogPage({ scope: 'all', limit: 20 });
const first = page.sessions[0];
if (first) {
  await hydrateSession({ source: first.source, sessionId: first.sessionId });
  const prompts = await getSession(first.sessionId);
  const events = await getSessionEventsPage(first.sessionId, { limit: 200 });
  for await (const event of sessionEvents(first.sessionId)) consume(event);
  const tools = await getSessionToolCallsPage(first.source, first.sessionId, { limit: 200 });
  const edits = await getSessionFileEditsPage(first.source, first.sessionId, { limit: 200 });
  for await (const call of sessionToolCalls(first.source, first.sessionId)) consume(call);
  for await (const edit of sessionFileEdits(first.source, first.sessionId)) consume(edit);
}
```

All APIs are async. `listSessionCatalog` and `listSessionCatalogPage` are
cache-only. `discoverSessions` is shallow discovery. `hydrateSession` is
targeted evidence acquisition for one existing catalog row. It returns
`hydrated`, `updated`, or `unchanged`, an indexed source stamp, evidence counts,
related native session IDs, and bounded-work metrics. `sync` is full ingestion.
Missing databases return empty read results; they do not trigger provider I/O.

Session discovery, listing, search, recent history, statistics, and sync accept
`scope: 'local' | 'remote' | 'all'`. Scope defaults to `local`, preserving
offline behavior and making provider-cloud access explicit. The CLI exposes the
same mutually exclusive `--local`, `--remote`, and `--all` flags; omitting them
is equivalent to `--local`.

Cached reads already support every scope. Remote acquisition runs through
provider connectors — claude.ai/code web sessions and Codex cloud tasks —
that are configured by the provider CLI's own sign-in on the machine (see the
repository's `docs/remote-connectors.md`). On a machine with no connector
configured, `discoverSessions({ scope: 'remote' })` and
`sync({ scope: 'remote' })` fail with `UnsupportedOperationError` and the stable
code `UNSUPPORTED_OPERATION`; they never fall back to local acquisition.
`scope: 'all'` runs the local adapters plus every configured connector.

Catalog pages, discovery results, statistics, and sync results echo the requested `scope`,
and discovery results additionally report `locationsRun` — the connector
locations that actually executed.
History and catalog rows have `locations`, containing `local`, `remote`, or both, so an
`all` query still returns one logical session while preserving where it was
found. `resumeCommand()` returns `null` for a remote-only history row rather
than emitting a local CLI command; an empty `locations` array retains legacy
local behavior for rows written before provenance tracking.

The event primitive is page-based and uses `{ tsMs, id }` as a deterministic
cursor. The `sessionEvents` async iterator walks pages without accumulating a
large transcript. `getSessionEvents` is an explicit collecting convenience.

Tool calls and file edits follow the same page / iterator / collect trio
(`getSessionToolCallsPage`, `sessionToolCalls`, `getSessionToolCalls`, and the
`...FileEdit...` equivalents), and take both a source and a session ID because
provider session IDs are not unique across providers. Their cursor is
`{ tsMs: number | null, id: number }`: a record may be indexed without a
timestamp, and undated records are ordered last. Stored provider JSON is parsed
into `args` and `structuredPatch`; a value that is absent or unparseable
becomes `null` while the raw string stays available as `argsJson` and
`structuredPatchJson`, so an unreadable payload never fails a page.

Native loading failures distinguish unsupported platforms, missing optional
platform packages, addon load failures, SDK/native contract mismatches, and
database open failures through stable `RelayHistoryError` subclasses. Provider
capability failures use `UnsupportedOperationError`.

The old synchronous `AiHist` class and `openAiHist()` API were removed in 1.0.
See [the migration guide](https://github.com/AgentWorkforce/relayhistory/blob/main/docs/native-sdk-migration.md).
