import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import initSqlJs from 'sql.js';
import {
  SESSION_CATALOG_CONTRACT_VERSION,
  discoverSessions,
  openAiHist,
  type CatalogSession,
} from './index.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NEW_SHAPE_DDL = `CREATE TABLE sessions (
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
)`;

/** The pre-catalog `sessions` table, as an ai-hist 0.5.0 database has it. */
const OLD_SHAPE_DDL = `CREATE TABLE sessions (
  session_id TEXT NOT NULL,
  source TEXT NOT NULL,
  cwd TEXT,
  git_branch TEXT,
  first_activity_ms INTEGER,
  last_activity_ms INTEGER,
  last_assistant_text TEXT,
  raw_path TEXT,
  parser_version INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (session_id, source)
)`;

const HISTORY_DDL = `CREATE TABLE history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  session_id TEXT,
  project TEXT,
  prompt TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  git_branch TEXT
)`;

async function writeDb(dbPath: string, build: (db: import('sql.js').Database) => void): Promise<void> {
  const SQL = await initSqlJs();
  const db = new SQL.Database();
  try {
    db.run(HISTORY_DDL);
    build(db);
    await writeFile(dbPath, Buffer.from(db.export()));
  } finally {
    db.close();
  }
}

/** A catalog covering every quirk the native contract calls out. */
async function writeCatalogDb(dbPath: string): Promise<void> {
  await writeDb(dbPath, (db) => {
    db.run(NEW_SHAPE_DDL);
    const insert = db.prepare(
      `INSERT INTO sessions (session_id, source, cwd, git_branch, first_activity_ms,
         last_activity_ms, last_assistant_text, raw_path, parser_version, first_prompt,
         models_json, originator, agent_version, repo_url, initial_commit,
         workspace_roots_json, source_stamp, discovery_state)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    try {
      // Newest. Fully ingested, with models and workspace roots.
      insert.run([
        'codex-1', 'codex', '/work/api', 'dev', 3_000, 5_000, 'done', '/raw/codex-1.jsonl', 2,
        'add a retry', '["gpt-5.4","gpt-5.4-mini"]', 'codex_cli_rs', '0.148.0',
        'git@github.com:acme/api.git', 'abc123', '["/work/api","/work/shared"]',
        'v1:mtime-1', 'full',
      ]);
      // Shallow-only: no last_assistant_text, no models.
      insert.run([
        'claude-1', 'claude', '/work/app', 'main', 1_000, 4_000, null, '/raw/claude-1.jsonl', 1,
        'first human prompt', null, null, '1.2.3', null, null, null, 'v1:mtime-2', 'shallow',
      ]);
      // Cursor: never a first_activity_ms; last_activity_ms is mtime-derived.
      insert.run([
        'cursor-1', 'cursor', '/work/app', null, null, 2_000, null, '/raw/cursor-1.db', 1,
        null, 'not json at all', null, null, null, null, null, 'v1:mtime-3', 'shallow',
      ]);
      // Relay: no cwd, and a legacy NULL discovery_state.
      insert.run([
        'relay-1', 'relay', null, null, 500, 1_000, 'relay reply', null, 1,
        'relay thread', null, null, null, null, null, null, null, null,
      ]);
      // No timestamps at all — must sort last, and must not surface via beforeMs.
      insert.run([
        'grok-1', 'grok', '/work/app', null, null, null, null, null, 1,
        null, null, null, null, null, null, null, null, 'shallow',
      ]);
      // Trajectories are derived records, not sessions: never listed.
      insert.run([
        'traj-1', 'trajectory', '/work/app', null, 9_000, 9_000, null, null, 1,
        null, null, null, null, null, null, null, null, 'full',
      ]);
    } finally {
      insert.free();
    }
  });
}

function ids(sessions: CatalogSession[]): string[] {
  return sessions.map((session) => session.sessionId);
}

async function withCatalog(
  prefix: string,
  write: (dbPath: string) => Promise<void>,
  body: (dbPath: string) => Promise<void>,
): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  const dbPath = join(dir, 'history.db');
  try {
    await write(dbPath);
    await body(dbPath);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// listSessionCatalog
// ---------------------------------------------------------------------------

test('listSessionCatalog orders by recency, excludes trajectories, and parses JSON columns', async () => {
  await withCatalog('ai-hist-catalog-', writeCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      const rows = hist.listSessionCatalog();
      assert.deepEqual(ids(rows), ['codex-1', 'claude-1', 'cursor-1', 'relay-1', 'grok-1']);
      assert.ok(!rows.some((row) => row.source === 'trajectory'));

      const codex = rows[0]!;
      assert.deepEqual(codex, {
        source: 'codex',
        sessionId: 'codex-1',
        cwd: '/work/api',
        gitBranch: 'dev',
        firstActivityMs: 3_000,
        lastActivityMs: 5_000,
        firstPrompt: 'add a retry',
        lastAssistantText: 'done',
        models: ['gpt-5.4', 'gpt-5.4-mini'],
        originator: 'codex_cli_rs',
        agentVersion: '0.148.0',
        repoUrl: 'git@github.com:acme/api.git',
        initialCommit: 'abc123',
        workspaceRoots: ['/work/api', '/work/shared'],
        rawPath: '/raw/codex-1.jsonl',
        sourceStamp: 'v1:mtime-1',
        discoveryState: 'full',
        fromCache: true,
        parserVersion: 2,
      });

      // NULL and malformed JSON columns both mean "nothing observed".
      const claude = rows[1]!;
      assert.deepEqual(claude.models, []);
      assert.deepEqual(claude.workspaceRoots, []);
      assert.equal(claude.lastAssistantText, null);
      assert.equal(claude.discoveryState, 'shallow');
      assert.deepEqual(rows[2]!.models, []);

      // Provider quirks: cursor has no first activity, relay has no cwd.
      assert.equal(rows[2]!.firstActivityMs, null);
      assert.equal(rows[3]!.cwd, null);
      // A NULL discovery_state predates the catalog and reads as 'full'.
      assert.equal(rows[3]!.discoveryState, 'full');
      // Rows with no last activity sort last.
      assert.equal(rows[4]!.lastActivityMs, null);
    } finally {
      hist.close();
    }
  });
});

test('listSessionCatalog filters by source and limit, and paginates with beforeMs', async () => {
  await withCatalog('ai-hist-catalog-filter-', writeCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      assert.deepEqual(ids(hist.listSessionCatalog({ sources: ['claude'] })), ['claude-1']);
      assert.deepEqual(ids(hist.listSessionCatalog({ sources: ['claude', 'codex'] })), [
        'codex-1',
        'claude-1',
      ]);
      // A trajectory filter cannot smuggle trajectory rows back in.
      assert.deepEqual(hist.listSessionCatalog({ sources: ['trajectory'] }), []);

      assert.deepEqual(ids(hist.listSessionCatalog({ limit: 2 })), ['codex-1', 'claude-1']);

      const page1 = hist.listSessionCatalog({ limit: 2 });
      const page2 = hist.listSessionCatalog({ limit: 2, beforeMs: page1[1]!.lastActivityMs! });
      assert.deepEqual(ids(page2), ['cursor-1', 'relay-1']);
      // Keyset pagination never resurfaces null-timestamp rows.
      assert.deepEqual(ids(hist.listSessionCatalog({ beforeMs: 1_000 })), []);
    } finally {
      hist.close();
    }
  });
});

test('listSessionCatalog reads a pre-catalog database through the in-memory migration', async () => {
  const writeOld = (dbPath: string) =>
    writeDb(dbPath, (db) => {
      db.run(OLD_SHAPE_DDL);
      db.run(
        `INSERT INTO sessions (session_id, source, cwd, git_branch, first_activity_ms,
           last_activity_ms, last_assistant_text, raw_path, parser_version)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ['legacy-1', 'claude', '/work/app', 'main', 1_000, 2_000, 'older reply', '/raw/legacy.jsonl', 1],
      );
    });

  await withCatalog('ai-hist-catalog-legacy-', writeOld, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      const rows = hist.listSessionCatalog();
      assert.equal(rows.length, 1);
      const row = rows[0]!;
      // Old columns still read.
      assert.equal(row.sessionId, 'legacy-1');
      assert.equal(row.cwd, '/work/app');
      assert.equal(row.lastAssistantText, 'older reply');
      // Columns the ALTER added exist and read as "nothing observed".
      assert.equal(row.firstPrompt, null);
      assert.equal(row.originator, null);
      assert.deepEqual(row.models, []);
      assert.deepEqual(row.workspaceRoots, []);
      assert.equal(row.sourceStamp, null);
      // A row written before discovery existed is a fully ingested row.
      assert.equal(row.discoveryState, 'full');
    } finally {
      hist.close();
    }
  });
});

test('a project scope constrains the catalog listing by cwd', async () => {
  await withCatalog('ai-hist-catalog-scope-', writeCatalogDb, async (dbPath) => {
    const scoped = await openAiHist({ dbPath, projectScope: '/work/app' });
    try {
      // /work/api is outside the scope; relay has no cwd, so it drops out too.
      assert.deepEqual(ids(scoped.listSessionCatalog()), ['claude-1', 'cursor-1', 'grok-1']);
    } finally {
      scoped.close();
    }
  });
});

test('the catalog is empty in JSONL fallback mode', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'ai-hist-catalog-fallback-'));
  const previousDb = process.env.AI_HIST_DB;
  const previousTrajectory = process.env.TRAJECTORY_ROOT;
  const previousOpenCode = process.env.OPENCODE_DB;
  process.env.AI_HIST_DB = join(dir, 'missing.db');
  process.env.TRAJECTORY_ROOT = join(dir, 'missing-trajectories');
  process.env.OPENCODE_DB = join(dir, 'missing-opencode.db');
  try {
    const hist = await openAiHist({ dbPath: join(dir, 'missing.db') });
    try {
      assert.equal(hist.sourceKind, 'jsonl');
      assert.deepEqual(hist.listSessionCatalog(), []);
    } finally {
      hist.close();
    }
  } finally {
    if (previousDb === undefined) delete process.env.AI_HIST_DB;
    else process.env.AI_HIST_DB = previousDb;
    if (previousTrajectory === undefined) delete process.env.TRAJECTORY_ROOT;
    else process.env.TRAJECTORY_ROOT = previousTrajectory;
    if (previousOpenCode === undefined) delete process.env.OPENCODE_DB;
    else process.env.OPENCODE_DB = previousOpenCode;
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// discoverSessions
// ---------------------------------------------------------------------------

/** Minimal fake child process that scripts stdout/stderr/exit for one run. */
function fakeSpawn(script: { chunks?: string[]; stderr?: string; code?: number; errorCode?: string }) {
  const calls: Array<{ bin: string; args: string[] }> = [];
  const spawnFn = ((bin: string, args: string[]) => {
    calls.push({ bin, args });
    const child = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter };
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    setImmediate(() => {
      if (script.errorCode) {
        const err = new Error('spawn failed') as NodeJS.ErrnoException;
        err.code = script.errorCode;
        child.emit('error', err);
        return;
      }
      for (const chunk of script.chunks ?? []) child.stdout.emit('data', Buffer.from(chunk, 'utf8'));
      if (script.stderr) child.stderr.emit('data', script.stderr);
      child.emit('close', script.code ?? 0);
    });
    return child;
  }) as unknown as typeof import('node:child_process').spawn;
  return { spawnFn, calls };
}

const DISCOVER_SESSION_LINE = JSON.stringify({
  type: 'session',
  source: 'codex',
  session_id: 'codex-cli',
  cwd: '/work/api',
  git_branch: 'dev',
  first_activity_ms: 1_000,
  last_activity_ms: 2_000,
  first_prompt: 'add a retry',
  last_assistant_text: null,
  models: ['gpt-5.4'],
  originator: 'codex_cli_rs',
  agent_version: '0.148.0',
  repo_url: null,
  initial_commit: null,
  workspace_roots: [],
  raw_path: '/raw/codex.jsonl',
  source_stamp: 'v1:stamp',
  discovery_state: 'shallow',
  from_cache: false,
});

const DISCOVER_SUMMARY_LINE = JSON.stringify({
  type: 'summary',
  contract_version: 1,
  discovered: 1,
  skipped_unchanged: 1,
  providers: {
    codex: { candidates: 2, discovered: 1, skipped_unchanged: 1, failed: false },
    grok: { candidates: 0, discovered: 0, skipped_unchanged: 0, failed: true },
  },
  exempt_sources: [{ source: 'trajectory', reason: 'derived trajectory records, not provider sessions' }],
  counters: {
    candidates_enumerated: 2,
    shallow_reads: 1,
    skipped_unchanged: 1,
    files_opened: 1,
    bytes_read: 4096,
  },
});

const DISCOVER_DIAGNOSTIC_LINE = JSON.stringify({
  type: 'diagnostic',
  source: 'grok',
  locator: '/raw/broken.json',
  error: 'unreadable',
});

test('discoverSessions parses the JSONL stream into sessions, diagnostics, and a summary', async () => {
  // Split mid-line so the incremental line buffering is exercised.
  const stream = `${DISCOVER_SESSION_LINE}\n${DISCOVER_DIAGNOSTIC_LINE}\n${DISCOVER_SUMMARY_LINE}\n`;
  const split = Math.floor(DISCOVER_SESSION_LINE.length / 2);
  const { spawnFn, calls } = fakeSpawn({ chunks: [stream.slice(0, split), stream.slice(split)] });

  const streamed: string[] = [];
  const result = await discoverSessions({
    binPath: '/bin/echo',
    spawnFn,
    sources: ['codex', 'grok'],
    limit: 5,
    onSession: (session) => streamed.push(session.sessionId),
  });

  assert.deepEqual(calls[0]!.args, [
    'sessions',
    'discover',
    '--json',
    '--source',
    'codex',
    '--source',
    'grok',
    '--limit',
    '5',
  ]);
  assert.deepEqual(streamed, ['codex-cli']);
  assert.equal(result.sessions.length, 1);
  assert.deepEqual(result.sessions[0], {
    source: 'codex',
    sessionId: 'codex-cli',
    cwd: '/work/api',
    gitBranch: 'dev',
    firstActivityMs: 1_000,
    lastActivityMs: 2_000,
    firstPrompt: 'add a retry',
    lastAssistantText: null,
    models: ['gpt-5.4'],
    originator: 'codex_cli_rs',
    agentVersion: '0.148.0',
    repoUrl: null,
    initialCommit: null,
    workspaceRoots: [],
    rawPath: '/raw/codex.jsonl',
    sourceStamp: 'v1:stamp',
    discoveryState: 'shallow',
    fromCache: false,
  });
  assert.deepEqual(result.diagnostics, [
    { source: 'grok', locator: '/raw/broken.json', error: 'unreadable' },
  ]);
  assert.equal(result.summary?.contractVersion, SESSION_CATALOG_CONTRACT_VERSION);
  assert.equal(result.summary?.discovered, 1);
  assert.equal(result.summary?.skippedUnchanged, 1);
  assert.deepEqual(result.summary?.providers.codex, {
    candidates: 2,
    discovered: 1,
    skippedUnchanged: 1,
    failed: false,
  });
  assert.equal(result.summary?.providers.grok?.failed, true);
  assert.deepEqual(result.summary?.exemptSources, [
    { source: 'trajectory', reason: 'derived trajectory records, not provider sessions' },
  ]);
  assert.deepEqual(result.summary?.counters, {
    candidatesEnumerated: 2,
    shallowReads: 1,
    skippedUnchanged: 1,
    filesOpened: 1,
    bytesRead: 4096,
  });
});

test('discoverSessions skips unparseable and unknown lines instead of failing', async () => {
  const { spawnFn, calls } = fakeSpawn({
    chunks: [
      `{ truncated\n`,
      `${JSON.stringify({ type: 'future_line_type', hello: 'world' })}\n`,
      `\n`,
      // No trailing newline: the last line still has to be parsed on close.
      `${DISCOVER_SESSION_LINE}\n${DISCOVER_SUMMARY_LINE}`,
    ],
  });
  const result = await discoverSessions({ binPath: '/bin/echo', spawnFn });
  assert.deepEqual(calls[0]!.args, ['sessions', 'discover', '--json']);
  assert.deepEqual(
    result.sessions.map((session) => session.sessionId),
    ['codex-cli'],
  );
  assert.equal(result.summary?.discovered, 1);
});

test('discoverSessions resolves with diagnostics when a provider fails but the run exits 0', async () => {
  const { spawnFn } = fakeSpawn({
    chunks: [`${DISCOVER_DIAGNOSTIC_LINE}\n${DISCOVER_SUMMARY_LINE}\n`],
    code: 0,
  });
  const seen: string[] = [];
  const result = await discoverSessions({
    binPath: '/bin/echo',
    spawnFn,
    onDiagnostic: (diagnostic) => seen.push(diagnostic.source),
  });
  assert.deepEqual(result.sessions, []);
  assert.deepEqual(seen, ['grok']);
  assert.equal(result.summary?.providers.grok?.failed, true);
});

test('discoverSessions rejects with a clear install hint when the binary is missing', async () => {
  const { spawnFn } = fakeSpawn({ errorCode: 'ENOENT' });
  await assert.rejects(
    discoverSessions({ binPath: '/nope/ai-hist', spawnFn }),
    /could not run the ai-hist binary.*AI_HIST_RUST_BIN/s,
  );
});

test('discoverSessions rejects on a non-zero exit and surfaces stderr', async () => {
  const { spawnFn } = fakeSpawn({ code: 2, stderr: 'every provider failed' });
  await assert.rejects(
    discoverSessions({ binPath: '/bin/echo', spawnFn }),
    /ai-hist sessions discover failed \(exit 2\).*every provider failed/,
  );
});

test('discoverSessions returns a null summary when the binary emits none', async () => {
  const { spawnFn } = fakeSpawn({ chunks: [`${DISCOVER_SESSION_LINE}\n`] });
  const result = await discoverSessions({ binPath: '/bin/echo', spawnFn });
  assert.equal(result.summary, null);
  assert.equal(result.sessions.length, 1);
});
