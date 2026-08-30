# Getting started

Install Node.js 20 or 22, then install RelayHistory from npm:

```bash
npm install --global ai-hist
ai-hist sessions discover --limit 100
ai-hist sessions list --limit 100
```

npm selects the matching prebuilt `ai-hist-native-*` package. Rust, Cargo, a
C/C++ compiler, curl installers, and separately installed binaries are not
part of normal installation.

The default database is
`~/.local/share/ai-hist/ai-history.db`. Set `AI_HIST_DB` to override it.

## Discovery, listing, and sync

These are intentionally different:

- `sessions list` is cache-only and never reads provider files.
- `sessions discover` performs bounded shallow reads and refreshes catalog
  metadata.
- `sync` performs full ingestion for search and event history.

```bash
ai-hist sessions discover --limit 100
ai-hist sessions list --limit 20
ai-hist sync
ai-hist search "authentication" --limit 20
ai-hist events SESSION_ID --limit 200
```

A missing database produces an empty list/search result. Reads do not trigger
hidden provider work.

## SDK

```bash
npm install ai-hist
```

```ts
import { discoverSessions, listSessionCatalog, search } from 'ai-hist';

await discoverSessions({ limit: 100 });
const sessions = await listSessionCatalog({ limit: 100 });
const matches = await search('authentication');
```

## MCP

```bash
npx -y ai-hist-mcp
```

If native loading fails, reinstall with npm optional dependencies enabled.
Errors state whether the platform is unsupported, a supported platform package
is missing, the addon failed to load, or package contract versions do not
match. RelayHistory never falls back to a different implementation.
