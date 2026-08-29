# ai-hist (TypeScript SDK and MCP server)

TypeScript reader for the [ai-hist](../README.md) history database, plus an MCP server for searching local AI coding-agent history from Claude Code, Codex, Cursor, Grok, and Agent Relay.

The SDK uses `sql.js`, so it has no native build step. It reads the same SQLite file that `ai-hist sync` writes, or falls back to scanning local Claude/Codex/Cursor/Grok JSONL files and compacted trajectory JSON when the database is missing.

## Install

```bash
npm install ai-hist
```

## Quick start

```ts
import { openAiHist, resumeCommand } from 'ai-hist';

const hist = await openAiHist(); // uses $AI_HIST_DB or ~/.local/share/ai-hist/ai-history.db
try {
  const sessions = hist.listSessions({ limit: 20 });
  for (const s of sessions) {
    console.log(`[${s.source}] ${s.firstPrompt.slice(0, 60)} (${s.promptCount} prompts)`);
    console.log('  resume:', resumeCommand(s));
  }
} finally {
  hist.close();
}
```

To require the ai-hist SQLite database instead of JSONL fallback:

```ts
const hist = await openAiHist({ fallback: 'error' });
```

## MCP server

After the package is published, run the local stdio MCP server with:

```bash
npx -y ai-hist-mcp
```

From a local checkout:

```bash
npm install
npm run build
node dist/mcp-server.js
```

The MCP server exposes tools for search, recent history, the session catalog, session lookup, temporal context, evidence packing, stats, trajectory search, and task WHY lookup over stdio. It runs on the user's machine and uses the same data-opening behavior as the SDK: SQLite first, then local fallback scanning.

To expose only one project through MCP, pass a project scope when launching the server. Exact project matches and child paths are included.

```bash
npx -y ai-hist-mcp --project .
npx -y ai-hist-mcp --project /path/to/project
```

Contract tools:

- `search_history(query, limit?)`
- `recent_entries(limit?, project?)`
- `get_session(session_id)`
- `get_context(id)`
- `stats()`
- `search_trajectories(query, limit?)`
- `why_for_task(query)`
- `list_sessions(sources?, limit?, before_ms?, after?)`

## Session catalog

`sessions` is a materialized catalog of every coding-agent session ai-hist knows about — one row per `(source, session_id)`, with cwd, git branch, first/last activity, the first prompt, observed models, originator, agent version, and repo identity. It is populated by shallow discovery (cheap, metadata only) and upgraded in place by a full `ai-hist sync`. `discoveryState` says which you are looking at: `'shallow'` for a catalog row whose transcript has not been ingested, `'full'` once it has (rows written before the catalog existed store `NULL` and read as `'full'`).

Reading it is one indexed query over a single table — no provider transcript is opened and no prompt history is scanned — so it is the fast path for "which sessions exist?" on first paint:

```ts
import { openAiHist, discoverSessions, SESSION_CATALOG_CONTRACT_VERSION } from 'ai-hist';

const hist = await openAiHist();
for (const s of hist.listSessionCatalog({ limit: 20 })) {
  console.log(s.lastActivityMs, s.source, s.sessionId, s.cwd, s.firstPrompt);
}
```

The catalog's total order is `(lastActivityMs DESC, source ASC, sessionId ASC)`, with rows of unknown recency last. Recency alone is not a key: one discovery pass can stamp many sessions with the same mtime-derived millisecond, so paginate with `listSessionCatalogPage` and follow its cursor, which carries all three columns:

```ts
let after;
for (;;) {
  const page = hist.listSessionCatalogPage({ limit: 20, after });
  render(page.sessions);
  if (!page.nextCursor) break;   // null once the catalog is exhausted
  after = page.nextCursor;       // { lastActivityMs, source, sessionId }
}
```

`nextCursor` is non-null only when the page filled its limit, so the loop terminates on an empty catalog and on a null-timestamp tail alike — both cases where a `beforeMs`-only walk either spins forever or silently drops the rows tied with a page boundary. `beforeMs` survives as a *coarse* cutoff ("anything before last Tuesday") and is ignored when `after` is set. The same contract is available from the CLI as the `next_cursor` field of `sessions list --json`, fed back as `--after-source` + `--after-session-id` (both required together, and `--after-ms` alone is refused — a timestamp is not a cursor). Omit `--after-ms` to continue through the undated tail.

`trajectory` rows are excluded — trajectories are derived records, not sessions. A configured `projectScope` constrains rows by `cwd`, so sources with no working directory (relay) drop out of a scoped listing. A negative `limit` throws (SQLite would read it as "unlimited"). In JSONL fallback mode (no SQLite database) the catalog is empty and the methods return no rows; only the native discovery engine writes it.

To populate or refresh the catalog, run shallow discovery. It scans the known provider locations with bounded reads, orders candidates globally by recency before applying the limit, and skips sources whose bytes have not changed since the last run:

```ts
const { sessions, diagnostics, summary } = await discoverSessions({
  sources: ['claude', 'codex'],   // omit for every discoverable source
  limit: 50,                      // global across providers, by recency
  onSession: (s) => render(s),    // streams as rows arrive
});

console.log(summary.discovered, summary.skippedUnchanged, summary.counters.bytesRead);
for (const d of diagnostics) console.warn(`[${d.source}] ${d.locator ?? ''}: ${d.error}`);
```

`discoverSessions` is a top-level function, not an `AiHist` method: it drives `ai-hist sessions discover --json` (binary discovery is `$AI_HIST_RUST_BIN` → the install.sh location → `ai-hist` on `PATH`, the same as `pushToCloud`), and that writes the on-disk database. An `AiHist` instance is an in-memory snapshot taken at open time, so re-open it before listing to see freshly discovered rows.

Failure behavior matches the CLI's: one provider blowing up yields a diagnostic and the run still resolves. The promise rejects with a `DiscoveryError` when the binary cannot be run at all (the message names `AI_HIST_RUST_BIN`), when the run exits non-zero because every selected provider failed, or when the closing summary is missing or announces a contract version this SDK does not implement — the version is checked for you against `SESSION_CATALOG_CONTRACT_VERSION` rather than left as a field to remember. The native command writes its diagnostics and summary trailer even on an all-provider failure, so a `DiscoveryError` carries `diagnostics`, `summary`, `stderr`, and `exitCode`. Unparseable or unrecognized JSONL lines are skipped rather than failing the run.

An exception thrown by `onSession` or `onDiagnostic` aborts the run: the child process is killed and the promise rejects with your error, instead of escaping the stream handler as an uncaught exception in the host.

## API

```ts
openAiHist(opts?: {
  dbPath?: string;
  projectScope?: string;
  fallback?: 'jsonl' | 'error';
}): Promise<AiHist>

hist.close(): void
hist.dbPath: string
hist.sourceKind: 'sqlite' | 'jsonl'
hist.projectScope: string | undefined

hist.recent(opts?): HistoryEntry[]            // newest prompts first
hist.listSessions(opts?): SessionSummary[]    // grouped from history by session_id, last activity DESC
hist.listSessionCatalog(opts?): CatalogSession[]      // the sessions catalog: cache-only, one indexed query
hist.listSessionCatalogPage(opts?): SessionCatalogPage // the same page plus its nextCursor
hist.getSession(sessionId): HistoryEntry[]    // all prompts in a session, oldest first
hist.getSessionEvents(sessionId): SessionEvent[] // full transcript: text, thinking, tool calls/results, token usage
hist.getToolCalls(sessionId): SessionToolCall[]  // the session's tool invocations
hist.getEntry(id): HistoryEntry | null
hist.getInTimeWindow(timestampMs, windowMs): HistoryEntry[]
hist.search(query, opts?): HistoryEntry[]     // literal substring search, recent matches first
hist.searchTrajectories(query, opts?): TrajectoryEntry[]
hist.whyForTask(query): TrajectoryEntry | null
hist.stats(): Stats                           // counts + date range
```

All list-style methods accept `{ source?, project?, limit?, beforeMs? }`. `beforeMs` is the cursor for paginating older results.

The catalog methods take `{ sources?, limit?, beforeMs?, after? }` instead — they filter on the catalog's own `source` column, not on project, and `after` is the cursor for paging.

```ts
resumeCommand(entry): string | null           // shell command per source; null for relay
defaultDbPath(): string                       // resolve env / OS default
discoverSessions(opts?): Promise<DiscoverResult>  // drives `ai-hist sessions discover --json`
SESSION_CATALOG_CONTRACT_VERSION: number      // native session-catalog output contract (1)
DiscoveryError                                // thrown by discoverSessions; carries diagnostics + summary
```

## Trajectories

ai-hist indexes compacted per-run trajectory JSON files as the decision WHY. Set `TRAJECTORY_ROOT` to an explicit root; the scanner reads:

```text
$TRAJECTORY_ROOT/**/compacted/*.json
```

Without `TRAJECTORY_ROOT`, default discovery scans:

```text
~/Projects/**/.trajectories/**/compacted/*.json
```

The runtime contract is one JSON file per completed run:

```json
{
  "id": "run-id",
  "version": 1,
  "personaId": "planner",
  "projectId": "agent-workforce",
  "task": { "title": "Task title", "description": "Task description" },
  "status": "completed",
  "startedAt": "2026-06-06T10:00:00.000Z",
  "completedAt": "2026-06-06T10:05:00.000Z",
  "decisions": [
    {
      "question": "What should we do?",
      "chosen": "Chosen option",
      "reasoning": "Why this option won",
      "alternatives": ["Other option"]
    }
  ],
  "retrospective": {
    "summary": "What happened",
    "approach": "How the work was done",
    "learnings": ["What to carry forward"],
    "confidence": 0.8
  }
}
```

## Schema

The canonical ai-hist SQLite schema is:

```sql
CREATE TABLE history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  session_id TEXT,
  project TEXT,
  prompt TEXT NOT NULL,
  prompt_hash TEXT,
  timestamp_ms INTEGER NOT NULL,
  UNIQUE(source, timestamp_ms, prompt)
);

CREATE VIRTUAL TABLE history_fts USING fts5(prompt, project, content='history', content_rowid='id');

CREATE TABLE sessions (
  session_id TEXT NOT NULL,
  source TEXT NOT NULL,
  cwd TEXT,
  git_branch TEXT,
  first_activity_ms INTEGER,
  last_activity_ms INTEGER,
  last_assistant_text TEXT,
  raw_path TEXT,
  parser_version INTEGER NOT NULL DEFAULT 1,
  first_prompt TEXT,
  models_json TEXT,
  originator TEXT,
  agent_version TEXT,
  repo_url TEXT,
  initial_commit TEXT,
  workspace_roots_json TEXT,
  source_stamp TEXT,
  discovery_state TEXT,
  PRIMARY KEY (session_id, source)
);

CREATE INDEX idx_sessions_recency ON sessions(last_activity_ms DESC, source, session_id);
CREATE INDEX idx_sessions_source_recency ON sessions(source, last_activity_ms DESC, session_id);
CREATE INDEX idx_sessions_source_last ON sessions(source, last_activity_ms DESC);
CREATE INDEX idx_sessions_raw_path ON sessions(source, raw_path);

CREATE TABLE discovery_skips (
  source TEXT NOT NULL,
  locator TEXT NOT NULL,
  stamp TEXT NOT NULL,
  reason TEXT,
  updated_ms INTEGER,
  PRIMARY KEY (source, locator)
);
```

`models_json` and `workspace_roots_json` hold JSON string arrays or `NULL` (never `[]`); the SDK parses both into arrays. The catalog columns were added after 0.5.0, so when the SDK opens an older database file it backfills the missing columns on its in-memory copy with `ALTER TABLE` — the file on disk is not touched, and the absent values read as `NULL`.

Trajectory sync also creates a structured `trajectories` table and inserts each per-run compact file into `history` with `source='trajectory'`, so general history search and WHY-specific lookup both work.

## License

MIT
