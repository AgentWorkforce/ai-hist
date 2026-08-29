import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import initSqlJs from 'sql.js';
import {
  SESSION_CATALOG_CONTRACT_VERSION,
  DiscoveryError,
  discoverSessions,
  openAiHist,
  type CatalogCursor,
  type CatalogSession,
} from './index.js';

const execFileAsync = promisify(execFile);

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

/**
 * A catalog where one discovery pass stamped many sessions with the same
 * millisecond, plus an undated tail — the shape a recency-only cursor drops
 * rows from.
 */
async function writeTiedCatalogDb(dbPath: string): Promise<void> {
  await writeDb(dbPath, (db) => {
    db.run(NEW_SHAPE_DDL);
    const insert = db.prepare(
      `INSERT INTO sessions (session_id, source, last_activity_ms, discovery_state)
       VALUES (?, ?, ?, 'shallow')`,
    );
    try {
      const rows: Array<[string, string, number | null]> = [
        ['tie-a', 'claude', 2_000],
        ['tie-b', 'claude', 2_000],
        ['tie-c', 'codex', 2_000],
        ['tie-d', 'codex', 2_000],
        ['old-a', 'claude', 1_000],
        ['old-b', 'cursor', 1_000],
        ['undated-a', 'claude', null],
        ['undated-b', 'codex', null],
        ['undated-c', 'codex', null],
      ];
      for (const row of rows) insert.run(row);
    } finally {
      insert.free();
    }
  });
}

function ids(sessions: CatalogSession[]): string[] {
  return sessions.map((session) => session.sessionId);
}

/** Identity of a row in the catalog's total order. */
function keys(sessions: CatalogSession[]): string[] {
  return sessions.map((session) => `${session.source}:${session.sessionId}`);
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

test('listSessionCatalogPage walks tied and undated rows exactly once', async () => {
  await withCatalog('ai-hist-catalog-page-', writeTiedCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      const everything = hist.listSessionCatalog({ limit: 100 });
      assert.equal(everything.length, 9);
      // Total order: recency first, then source, then session id; undated last.
      assert.deepEqual(ids(everything), [
        'tie-a',
        'tie-b',
        'tie-c',
        'tie-d',
        'old-a',
        'old-b',
        'undated-a',
        'undated-b',
        'undated-c',
      ]);

      // A walk in pages of two must reproduce that list exactly: a page
      // boundary lands inside a group of tied rows, which is where a
      // recency-only cursor loses (or repeats) rows.
      const walked: string[] = [];
      let after: CatalogCursor | undefined;
      let pages = 0;
      for (;;) {
        const page = hist.listSessionCatalogPage({ limit: 2, after });
        walked.push(...keys(page.sessions));
        pages += 1;
        assert.ok(pages < 20, 'pagination did not terminate');
        if (!page.nextCursor) break;
        after = page.nextCursor;
      }
      assert.deepEqual(walked, keys(everything));
      assert.equal(new Set(walked).size, walked.length, 'a row was returned twice');
      assert.equal(pages, 5); // 2+2+2+2+1
    } finally {
      hist.close();
    }
  });
});

test('a dated cursor still reaches the undated tail', async () => {
  await withCatalog('ai-hist-catalog-tail-', writeTiedCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      const fromLastDated = hist.listSessionCatalog({
        after: { lastActivityMs: 1_000, source: 'cursor', sessionId: 'old-b' },
      });
      assert.deepEqual(ids(fromLastDated), ['undated-a', 'undated-b', 'undated-c']);

      // And an undated cursor continues within the tail, by identity alone.
      const withinTail = hist.listSessionCatalog({
        after: { lastActivityMs: null, source: 'claude', sessionId: 'undated-a' },
      });
      assert.deepEqual(ids(withinTail), ['undated-b', 'undated-c']);

      // A cursor sitting on a tie returns the rest of the tie, not the next
      // timestamp — the failure mode `before_ms` alone cannot avoid.
      const midTie = hist.listSessionCatalog({
        limit: 2,
        after: { lastActivityMs: 2_000, source: 'claude', sessionId: 'tie-b' },
      });
      assert.deepEqual(ids(midTie), ['tie-c', 'tie-d']);
    } finally {
      hist.close();
    }
  });
});

test('nextCursor is set only when a page fills its limit', async () => {
  await withCatalog('ai-hist-catalog-cursor-', writeTiedCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      // Short page: the catalog is exhausted, so there is nothing to continue.
      const short = hist.listSessionCatalogPage({ limit: 50 });
      assert.equal(short.sessions.length, 9);
      assert.equal(short.nextCursor, null);

      // Exactly-full page: the cursor is handed out even though the next page
      // turns out to be empty — the same fill rule the native page uses.
      const exact = hist.listSessionCatalogPage({ limit: 9 });
      assert.deepEqual(exact.nextCursor, {
        lastActivityMs: null,
        source: 'codex',
        sessionId: 'undated-c',
      });
      const beyond = hist.listSessionCatalogPage({ limit: 9, after: exact.nextCursor! });
      assert.deepEqual(beyond.sessions, []);
      assert.equal(beyond.nextCursor, null);

      // The cursor carries the last row of the page, ties included.
      const first = hist.listSessionCatalogPage({ limit: 2 });
      assert.deepEqual(first.nextCursor, {
        lastActivityMs: 2_000,
        source: 'claude',
        sessionId: 'tie-b',
      });
    } finally {
      hist.close();
    }
  });
});

test('a negative limit is rejected rather than dumping the catalog', async () => {
  await withCatalog('ai-hist-catalog-limit-', writeTiedCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      assert.throws(() => hist.listSessionCatalog({ limit: -1 }), RangeError);
      assert.throws(() => hist.listSessionCatalogPage({ limit: -1 }), RangeError);
    } finally {
      hist.close();
    }
  });
});

test('a half-built cursor is rejected rather than silently restarting the walk', async () => {
  await withCatalog('ai-hist-catalog-halfcursor-', writeTiedCatalogDb, async (dbPath) => {
    const hist = await openAiHist({ dbPath });
    try {
      // A timestamp on its own cannot separate tied rows; dropping it would
      // hand back page one forever.
      assert.throws(
        () => hist.listSessionCatalog({ after: { lastActivityMs: 2_000 } as never }),
        TypeError,
      );
      assert.throws(
        () => hist.listSessionCatalog({ after: { lastActivityMs: 2_000, source: 'claude' } as never }),
        TypeError,
      );
      assert.throws(
        () => hist.listSessionCatalog({ after: { source: '', sessionId: 'tie-a' } }),
        TypeError,
      );
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
  // The fallback scanner resolves ~/.claude, ~/.codex, ~/.cursor and ~/.grok
  // from os.homedir() at module load, so overriding HOME inside this process
  // would come too late and the scan would read the developer's real history.
  // Run it in a child with an empty HOME instead: fully isolated, and it also
  // proves the fallback path builds no catalog rather than merely finding none.
  const dir = await mkdtemp(join(tmpdir(), 'ai-hist-catalog-fallback-'));
  const home = join(dir, 'home');
  await mkdir(home, { recursive: true });
  const probe = join(dir, 'probe.mjs');
  const indexUrl = new URL('./index.js', import.meta.url).href;
  await writeFile(
    probe,
    `import { openAiHist } from ${JSON.stringify(indexUrl)};\n` +
      `const hist = await openAiHist({ dbPath: process.env.AI_HIST_DB });\n` +
      `console.log(JSON.stringify({ kind: hist.sourceKind, catalog: hist.listSessionCatalog() }));\n` +
      `hist.close();\n`,
  );
  try {
    const { stdout } = await execFileAsync(process.execPath, [probe], {
      env: {
        ...process.env,
        HOME: home,
        USERPROFILE: home,
        AI_HIST_DB: join(dir, 'missing.db'),
        TRAJECTORY_ROOT: join(dir, 'missing-trajectories'),
        OPENCODE_DB: join(dir, 'missing-opencode.db'),
      },
      timeout: 30_000,
    });
    const result = JSON.parse(stdout) as { kind: string; catalog: unknown[] };
    assert.equal(result.kind, 'jsonl');
    assert.deepEqual(result.catalog, []);
  } finally {
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

test('discoverSessions rejects on a non-zero exit, carrying the diagnostics and summary', async () => {
  // Every provider failed: the CLI still writes its diagnostics and the summary
  // trailer before exiting non-zero, so the error must keep them.
  const { spawnFn } = fakeSpawn({
    chunks: [`${DISCOVER_DIAGNOSTIC_LINE}\n${DISCOVER_SUMMARY_LINE}\n`],
    code: 2,
    stderr: 'every provider failed',
  });
  await assert.rejects(
    discoverSessions({ binPath: '/bin/echo', spawnFn }),
    (err: unknown) => {
      assert.ok(err instanceof DiscoveryError);
      assert.match(err.message, /ai-hist sessions discover failed \(exit 2\).*every provider failed/);
      assert.equal(err.exitCode, 2);
      assert.deepEqual(err.diagnostics.map((d) => d.source), ['grok']);
      assert.equal(err.summary?.providers.grok?.failed, true);
      return true;
    },
  );
});

test('discoverSessions rejects when the run produces no summary trailer', async () => {
  const { spawnFn } = fakeSpawn({ chunks: [`${DISCOVER_SESSION_LINE}\n`] });
  await assert.rejects(
    discoverSessions({ binPath: '/bin/echo', spawnFn }),
    /produced no summary line/,
  );
});

test('discoverSessions rejects a summary from an unsupported contract version', async () => {
  const future = JSON.stringify({
    ...(JSON.parse(DISCOVER_SUMMARY_LINE) as Record<string, unknown>),
    contract_version: 2,
  });
  const { spawnFn } = fakeSpawn({ chunks: [`${DISCOVER_SESSION_LINE}\n${future}\n`] });
  await assert.rejects(
    discoverSessions({ binPath: '/bin/echo', spawnFn }),
    (err: unknown) => {
      assert.ok(err instanceof DiscoveryError);
      assert.match(err.message, /unsupported session-catalog contract version 2/);
      // The rows the run did emit are still on the error's summary.
      assert.equal(err.summary?.discovered, 1);
      return true;
    },
  );

  // A summary with no contract_version at all is equally unusable.
  const versionless = JSON.stringify({ type: 'summary', discovered: 0 });
  const { spawnFn: spawnVersionless } = fakeSpawn({ chunks: [`${versionless}\n`] });
  await assert.rejects(
    discoverSessions({ binPath: '/bin/echo', spawnFn: spawnVersionless }),
    /unsupported session-catalog contract version 0/,
  );
});

test('an exception from onSession aborts the run instead of escaping the stream', async () => {
  const { spawnFn } = fakeSpawn({
    chunks: [`${DISCOVER_SESSION_LINE}\n${DISCOVER_SUMMARY_LINE}\n`],
  });
  const boom = new Error('host render failed');
  await assert.rejects(
    discoverSessions({
      binPath: '/bin/echo',
      spawnFn,
      onSession: () => {
        throw boom;
      },
    }),
    (err: unknown) => err === boom,
  );
});

test('an exception from onDiagnostic aborts the run too', async () => {
  const { spawnFn } = fakeSpawn({
    chunks: [`${DISCOVER_DIAGNOSTIC_LINE}\n${DISCOVER_SUMMARY_LINE}\n`],
  });
  const boom = new Error('host logger failed');
  await assert.rejects(
    discoverSessions({
      binPath: '/bin/echo',
      spawnFn,
      onDiagnostic: () => {
        throw boom;
      },
    }),
    (err: unknown) => err === boom,
  );
});
