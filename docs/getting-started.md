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

They use one session ledger. Local and remote describe where a session was
observed (its presence), not separate kinds of session or separate databases.
Commands that select a set of sessions accept exactly one of:

- `--local` — sessions with a local presence; this is the default.
- `--remote` — sessions with a remote provider presence.
- `--all` — the union, deduplicated so a session with both presences appears
  once.

Omitting the flag is exactly equivalent to `--local`. Combining scope flags is
an error.

```bash
ai-hist sessions discover --local --limit 100
ai-hist sessions list --all --limit 20
ai-hist sync --local
ai-hist search "authentication" --all --limit 20
ai-hist recent --remote --limit 20
ai-hist events SESSION_ID --limit 200
```

`sessions list`, `search`, and `recent` only query cached rows in the unified
ledger, regardless of scope. A missing database produces an empty result.
Reads do not trigger hidden provider work. `session` and `events` address an
already-known session directly, so they are scope-independent and do not
accept a location flag.

Remote acquisition runs through provider connectors: `claude-web` lists your
claude.ai/code web sessions with the OAuth sign-in the Claude Code CLI stored,
and `codex-cloud` lists Codex cloud tasks through `codex cloud list --json`.
A connector is configured when the provider CLI is signed in on this machine;
see [Remote connectors](remote-connectors.md) for detection, credentials, and
fidelity. With no connector configured, `sessions discover --remote` and
`sync --remote` fail explicitly instead of falling back to local work, while
`--all` runs local adapters plus whatever connectors are configured. Cached
remote/all queries are part of the stable contract either way.

Discovery results keep those two facts separate: the summary `scope` is the
scope requested by the caller, while row `locations` are the presences actually
observed. Thus a current `--all` discovery summary says `all` even though its
connector-location work is local-only.

## SDK

```bash
npm install ai-hist
```

```ts
import { discoverSessions, listSessionCatalog, search } from 'ai-hist';

await discoverSessions({ limit: 100, scope: 'local' });
const sessions = await listSessionCatalog({ limit: 100, scope: 'all' });
const matches = await search('authentication', { scope: 'all' });
```

## MCP

```bash
npx -y ai-hist-mcp
```

If native loading fails, reinstall with npm optional dependencies enabled.
Errors state whether the platform is unsupported, a supported platform package
is missing, the addon failed to load, or package contract versions do not
match. RelayHistory never falls back to a different implementation.
