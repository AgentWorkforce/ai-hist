# RelayHistory (`ai-hist`)

RelayHistory indexes coding-agent sessions from Claude Code, Codex, Cursor,
Grok, OpenCode, and Agent Relay in a local SQLite database.

Its production architecture has one implementation:

```text
Rust engine (providers + SQLite + migrations + queries)
  → Node-API addon
    → TypeScript SDK
      → Node CLI / MCP server
```

TypeScript never opens SQLite, scans provider files, or invokes another
executable. The native addon is mandatory and operates on the SQLite file in
place, including WAL-backed databases.

## Install

Node.js 20 or 22 is required. npm installs the SDK, CLI, MCP server, native
loader, and matching prebuilt platform package:

```bash
npm install ai-hist

# Global CLI
npm install --global ai-hist
ai-hist sessions discover --limit 100
ai-hist sessions list --limit 100
```

Session-set commands share one mutually exclusive location scope:

```bash
ai-hist sessions list --local   # default; identical to omitting a scope flag
ai-hist sessions list --remote  # cached sessions with a remote presence
ai-hist sessions list --all     # union of both, with each session returned once
```

The same `--local` / `--remote` / `--all` contract applies to discovery,
search, recent history, statistics, and sync. Local and remote are presences of a session,
not separate catalogs: every result comes from the same session ledger. Reads
only filter that cached ledger. Remote discovery and remote sync are reserved
for provider connectors and currently return an unsupported-operation error;
they never silently fall back to local work. An acquisition summary's `scope`
echoes the requested scope; it does not claim that connectors exist at every
location in that scope. In particular, `--all` currently runs local connectors
only. Each catalog row's `locations` array reports where that session was
actually observed.

No Rust toolchain, C/C++ compiler, standalone CLI, curl installer, or runtime
binary download is used.

### Checking your version

`ai-hist --version` prints the installed npm package version. In an interactive
terminal it also checks npm for a newer release, with a three-second timeout and
silent offline fallback. Suppress the notice with `--no-warning` or
`RELAYHISTORY_NO_UPDATE_CHECK=1`:

```bash
ai-hist --version
ai-hist --version --no-warning
```

## TypeScript

```ts
import {
  discoverSessions,
  listSessionCatalog,
  getSessionEventsPage,
  search,
  sync,
} from 'ai-hist';

await discoverSessions({ limit: 100, scope: 'local' });
const sessions = await listSessionCatalog({ limit: 100, scope: 'all' });
const firstEvents = sessions[0]
  ? await getSessionEventsPage(sessions[0].sessionId, {
      source: sessions[0].source,
      limit: 200,
    })
  : null;
const matches = await search('migration', { limit: 20, scope: 'all' });
await sync({ scope: 'local' }); // explicit full ingestion; local is the default
```

`listSessionCatalog` is cache-only. `discoverSessions` performs bounded shallow
provider discovery and updates the shared ledger. `sync` performs full
ingestion. Reads never silently turn into either discovery or sync. Direct
session and event lookup is identity-based and therefore has no location scope.

## MCP

```bash
npx -y ai-hist-mcp
```

MCP exposes thin adapters for search, recent history, catalog listing,
discovery, sessions, paged events, statistics, and explicit sync.

## Supported production matrix

| OS/runtime | Architectures |
|---|---|
| macOS 12+ | arm64, x64 |
| Linux glibc | arm64, x64 |
| Linux musl | arm64, x64 |
| Windows 10/11 and Server 2022 | x64 MSVC |

Node-API level 4 is used. CI tests Node.js 20 and 22. Windows arm64, FreeBSD,
Bun, Deno, browsers, Electron renderer processes, and other unlisted runtimes
are not supported. Windows arm64 remains follow-up work until it can be built
and executed reliably in CI.

See [architecture](docs/architecture.md), [getting started](docs/getting-started.md),
[migration](docs/native-sdk-migration.md), and [release validation](docs/releasing.md).

## Repository development

The `ai-hist-engine` Rust binary target remains an internal development harness while
provider-ingestion code is being physically separated from command rendering.
It is not published as a second user runtime. Normal users install with npm.

```bash
cargo test --workspace
cd crates/ai-hist-napi && npm ci && npm run build:debug
cd ../../sdk-ts && npm install && npm test
```
