# Migration to ai-hist 1.0

Version 1.0 is deliberately breaking. Remove standalone/curl-installed
`ai-hist` binaries and install the npm package globally if CLI access is
needed:

```bash
npm uninstall --global ai-hist-mcp
type -a ai-hist
npm install --global ai-hist ai-hist-mcp
type -a ai-hist
```

Before installing, remove any standalone executable reported by the first
`type -a` (commonly `~/.local/bin/ai-hist`). After installing, verify the
first result points into npm's global installation rather than the old binary.

The environment variables `AI_HIST_RUST_BIN` and `AI_HIST_CLI` no longer have
meaning. `AI_HIST_DB` remains the database-path override.

Replace the synchronous snapshot API:

```ts
// before
const hist = await openAiHist({ fallback: 'jsonl' });
const rows = hist.search('cursor');
hist.close();

// 1.0
const rows = await search('cursor');
```

Every database API is now async. Remove `fallback`, `sourceKind`, `close`, and
binary-path options. If the database is missing, read APIs return empty data;
call `discoverSessions()` or `sync()` explicitly.

Replace unbounded event reads with a page or async iterator:

```ts
const page = await getSessionEventsPage(id, { limit: 200 });
for await (const event of sessionEvents(id, { limit: 200 })) consume(event);
```

The 1.0 CLI covers `sessions list`, `sessions discover`, `search`, `recent`,
`session`, `events`, `stats`, and `sync`. Removed legacy cloud/Pair/tag/
trajectory convenience commands must migrate to dedicated services or future
native-backed SDK operations; they are not retained through subprocess or
JavaScript-SQL fallbacks.
