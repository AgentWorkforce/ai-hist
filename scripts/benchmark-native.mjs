import { execFile, spawn } from 'node:child_process';
import { appendFile, copyFile, mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises';
import { arch, cpus, platform, release, tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import {
  defaultDbPath, discoverSessions, getSessionEventsPage, listSessionCatalogPage,
  nativeBuildProfile,
} from '../sdk-ts/dist/index.js';

// A debug addon runs the same algorithms several times slower; its numbers
// measure the compiler, not RelayHistory.
const buildProfile = await nativeBuildProfile();
if (buildProfile !== 'release') {
  console.error(
    `The installed ai-hist-native addon is a ${buildProfile} build; benchmarks require release. `
    + 'Rebuild it with: npm run build --prefix crates/ai-hist-napi',
  );
  process.exit(1);
}

const execFileAsync = promisify(execFile);
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const invocationDirectory = process.cwd();

function option(name) {
  const prefix = `--${name}=`;
  const inline = process.argv.find((argument) => argument.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  if (index >= 0) return process.argv[index + 1];
  return process.env[`npm_config_${name.replaceAll('-', '_')}`];
}

function flag(name) {
  if (process.argv.includes(`--${name}`)) return true;
  const value = process.env[`npm_config_${name.replaceAll('-', '_')}`]?.toLowerCase();
  return value === '1' || value === 'true' || value === 'yes';
}

const dbPath = resolve(option('db') || process.env.AI_HIST_DB || defaultDbPath());
const outputOption = option('output');
const outputPath = outputOption ? resolve(invocationDirectory, outputOption) : null;
const pretty = flag('pretty');
const discoveryLimits = [20, 100, 1_000];

async function timed(name, operation) {
  const start = performance.now();
  const value = await operation();
  return { name, ms: Number((performance.now() - start).toFixed(2)), value };
}

async function mcpColdDiscovery(home, benchmarkDbPath) {
  const child = spawn(process.execPath, [join(repositoryRoot, 'sdk-ts/dist/mcp-server.js')], {
    cwd: repositoryRoot,
    env: { ...process.env, HOME: home, USERPROFILE: home, AI_HIST_DB: benchmarkDbPath },
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  let buffer = '';
  let nextId = 1;
  const waiting = new Map();
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    buffer += chunk;
    for (;;) {
      const newline = buffer.indexOf('\n');
      if (newline < 0) break;
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      const message = JSON.parse(line);
      if (message.id != null && waiting.has(message.id)) {
        waiting.get(message.id)(message);
        waiting.delete(message.id);
      }
    }
  });
  const request = (method, params) => new Promise((resolveRequest, reject) => {
    const id = nextId++;
    waiting.set(id, resolveRequest);
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    setTimeout(() => reject(new Error(`MCP ${method} timed out`)), 10_000).unref();
  });
  try {
    await request('initialize', {
      protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'benchmark', version: '1' },
    });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
    const measurement = await timed('MCP cold shallow discovery 20', () => request('tools/call', {
      name: 'discover_sessions', arguments: { sources: ['claude'], limit: 20 },
    }));
    if (measurement.value.error || measurement.value.result?.isError) {
      throw new Error(`MCP discovery failed: ${JSON.stringify(measurement.value)}`);
    }
    const text = measurement.value.result?.content?.find((item) => item.type === 'text')?.text;
    if (typeof text !== 'string') throw new Error('MCP discovery returned no text result');
    let payload;
    try { payload = JSON.parse(text); }
    catch (error) { throw new Error(`MCP discovery returned invalid JSON: ${error.message}`); }
    if (!Array.isArray(payload.sessions)) throw new Error('MCP discovery result has no sessions array');
    return { name: measurement.name, ms: measurement.ms, rows: payload.sessions.length };
  } finally {
    child.kill();
  }
}

async function createDiscoveryFixture(home, count) {
  const project = join(home, '.claude', 'projects', '-relayhistory-benchmark');
  await mkdir(project, { recursive: true });
  const baseTimestamp = Date.UTC(2026, 0, 1);
  for (let index = 0; index < count; index++) {
    const sessionId = `benchmark-${String(index).padStart(4, '0')}`;
    const timestampMs = baseTimestamp + index * 1_000;
    const timestamp = new Date(timestampMs).toISOString();
    const path = join(project, `${sessionId}.jsonl`);
    const records = [
      { sessionId, cwd: '/benchmark/project', gitBranch: 'main', type: 'user', message: { role: 'user', content: `benchmark prompt ${index}` }, timestamp },
      { sessionId, type: 'assistant', message: { role: 'assistant', model: 'benchmark-model', content: `benchmark response ${index}` }, timestamp },
    ];
    await writeFile(path, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`, 'utf8');
    await utimes(path, timestampMs / 1_000, timestampMs / 1_000);
  }
}

async function withEnvironment(overrides, operation) {
  const saved = {};
  for (const [key, value] of Object.entries(overrides)) {
    saved[key] = process.env[key];
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return await operation();
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

// OPENCODE_DB would point discovery past the fixture home at a real store.
const withHome = (home, operation) => withEnvironment(
  { HOME: home, USERPROFILE: home, OPENCODE_DB: undefined },
  operation,
);

async function discoverFixture(home, benchmarkDbPath, source, limit) {
  return withHome(home, () => discoverSessions({
    dbPath: benchmarkDbPath, sources: [source], limit,
  }));
}

async function changeDiscoveredSessions(sessions) {
  const timestamp = new Date().toISOString();
  const originals = [];
  for (const session of sessions) {
    if (!session.rawPath) continue;
    const metadata = await stat(session.rawPath);
    originals.push({
      path: session.rawPath,
      contents: await readFile(session.rawPath),
      accessedAt: metadata.atime,
      modifiedAt: metadata.mtime,
    });
    const record = {
      sessionId: session.sessionId,
      type: 'assistant',
      message: { role: 'assistant', model: 'benchmark-model', content: 'changed benchmark response' },
      timestamp,
    };
    await appendFile(session.rawPath, `${JSON.stringify(record)}\n`, 'utf8');
  }
  return originals;
}

async function restoreFixtureSessions(originals) {
  for (const original of originals) {
    await writeFile(original.path, original.contents);
    await utimes(original.path, original.accessedAt, original.modifiedAt);
  }
}

// The minimal subset of opencode's store that discovery reads: sessions carry
// the change stamp, and message/part rows feed the excerpt and model queries.
async function createOpencodeFixture(home, count) {
  const databasePath = join(home, '.local', 'share', 'opencode', 'opencode.db');
  await mkdir(dirname(databasePath), { recursive: true });
  const db = new DatabaseSync(databasePath);
  try {
    db.exec(
      'CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT, time_created INTEGER, time_updated INTEGER);'
      + 'CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);'
      + 'CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);'
      + 'CREATE INDEX session_time_updated_id_idx ON session(time_updated DESC, id);'
      + 'CREATE INDEX message_session_time_created_id_idx ON message(session_id, time_created, id);'
      + 'CREATE INDEX part_session_idx ON part(session_id);'
      + 'CREATE INDEX part_message_id_id_idx ON part(message_id, id);',
    );
    const insertSession = db.prepare('INSERT INTO session VALUES (?, ?, ?, ?)');
    const insertMessage = db.prepare('INSERT INTO message VALUES (?, ?, ?, ?)');
    const insertPart = db.prepare('INSERT INTO part VALUES (?, ?, ?, ?, ?)');
    const baseTimestamp = Date.UTC(2026, 0, 1);
    db.exec('BEGIN');
    for (let index = 0; index < count; index++) {
      const sessionId = `benchmark-oc-${String(index).padStart(4, '0')}`;
      const timestampMs = baseTimestamp + index * 1_000;
      insertSession.run(sessionId, '/benchmark/project', timestampMs, timestampMs);
      insertMessage.run(`m-${index}`, sessionId, timestampMs, '{"role":"user","modelID":"benchmark-model"}');
      insertPart.run(`p-${index}`, `m-${index}`, sessionId, timestampMs, JSON.stringify({ type: 'text', text: `benchmark prompt ${index}` }));
    }
    db.exec('COMMIT');
  } finally {
    db.close();
  }
  return databasePath;
}

// A changed opencode session is a bumped `time_updated` stamp plus a new
// assistant message, mirroring the appended record in the Claude fixture.
// Restoration is a byte copy of the pristine store, not reversed rows: a
// DELETE leaves the inserted pages on the freelist, which would make later
// cases read a physically different fixture than the first cold case.
function changeOpencodeSessions(databasePath, sessions) {
  const timestampMs = Date.now();
  const db = new DatabaseSync(databasePath);
  try {
    const bumpStamp = db.prepare('UPDATE session SET time_updated = ? WHERE id = ?');
    const insertMessage = db.prepare('INSERT INTO message VALUES (?, ?, ?, ?)');
    const insertPart = db.prepare('INSERT INTO part VALUES (?, ?, ?, ?, ?)');
    db.exec('BEGIN');
    for (const session of sessions) {
      bumpStamp.run(timestampMs, session.sessionId);
      insertMessage.run(`changed-m-${session.sessionId}`, session.sessionId, timestampMs, '{"role":"assistant","modelID":"benchmark-model"}');
      insertPart.run(`changed-p-${session.sessionId}`, `changed-m-${session.sessionId}`, session.sessionId, timestampMs, '{"type":"text","text":"changed benchmark response"}');
    }
    db.exec('COMMIT');
  } finally {
    db.close();
  }
}

function discoveryMeasurement(measurement) {
  return {
    name: measurement.name,
    ms: measurement.ms,
    rows: measurement.value.sessions.length,
    counters: measurement.value.counters,
  };
}

async function measureDiscoveryScaling({ prefix, source, home, temporary, change, restore }) {
  const coldResults = [];
  const unchangedResults = [];
  const changedResults = [];
  for (const limit of discoveryLimits) {
    const coldDb = join(temporary, `${source}-cold-${limit}.db`);
    coldResults.push(discoveryMeasurement(await timed(
      `${prefix}cold shallow discovery ${limit}`,
      () => discoverFixture(home, coldDb, source, limit),
    )));

    const unchangedDb = join(temporary, `${source}-unchanged-${limit}.db`);
    await discoverFixture(home, unchangedDb, source, limit);
    unchangedResults.push(discoveryMeasurement(await timed(
      `${prefix}unchanged shallow discovery ${limit}`,
      () => discoverFixture(home, unchangedDb, source, limit),
    )));

    const changedDb = join(temporary, `${source}-changed-${limit}.db`);
    const seeded = await discoverFixture(home, changedDb, source, limit);
    const originals = await change(seeded.sessions);
    try {
      changedResults.push(discoveryMeasurement(await timed(
        `${prefix}cold->changed shallow discovery ${limit}`,
        () => discoverFixture(home, changedDb, source, limit),
      )));
    } finally {
      await restore(originals);
    }
  }
  return [...coldResults, ...unchangedResults, ...changedResults];
}

const info = await stat(dbPath);
const results = [];
const temporary = await mkdtemp(join(tmpdir(), 'relayhistory-benchmark-'));
try {
  const fixtureHome = join(temporary, 'home');
  await createDiscoveryFixture(fixtureHome, Math.max(...discoveryLimits));
  const opencodeDb = await createOpencodeFixture(fixtureHome, Math.max(...discoveryLimits));
  const opencodePristine = join(temporary, 'opencode-pristine.db');
  await copyFile(opencodeDb, opencodePristine);

  results.push(...await measureDiscoveryScaling({
    prefix: '', source: 'claude', home: fixtureHome, temporary,
    change: changeDiscoveredSessions,
    restore: restoreFixtureSessions,
  }));
  results.push(...await measureDiscoveryScaling({
    prefix: 'opencode ', source: 'opencode', home: fixtureHome, temporary,
    change: (sessions) => changeOpencodeSessions(opencodeDb, sessions),
    restore: () => copyFile(opencodePristine, opencodeDb),
  }));

  const firstPage = await listSessionCatalogPage({ dbPath, limit: 20 });
  const candidate = firstPage.sessions.find((session) => session.discoveryState === 'full') ?? firstPage.sessions[0];
  if (!candidate) throw new Error(`No session is available in ${dbPath} for the event benchmarks`);
  await getSessionEventsPage(candidate.sessionId, { dbPath, source: candidate.source, limit: 200 });
  for (const limit of [20, 200]) {
    const events = await timed(`warm session events ${limit}`, () => getSessionEventsPage(candidate.sessionId, {
      dbPath, source: candidate.source, limit,
    }));
    results.push({ name: events.name, ms: events.ms, rows: events.value.events.length });
  }

  const cliDb = join(temporary, 'cli-cold.db');
  const cli = await timed('CLI startup + cold shallow discovery 20', () => execFileAsync(process.execPath, [
    join(repositoryRoot, 'sdk-ts/dist/cli.js'), 'sessions', 'discover', '--source', 'claude',
    '--db', cliDb, '--limit', '20', '--json',
  ], {
    env: { ...process.env, HOME: fixtureHome, USERPROFILE: fixtureHome },
    maxBuffer: 10 * 1024 * 1024,
  }));
  const cliRows = cli.value.stdout.trim().split('\n')
    .map((line) => JSON.parse(line))
    .filter((line) => line.type === 'session').length;
  results.push({ name: cli.name, ms: cli.ms, rows: cliRows });

  results.push(await mcpColdDiscovery(fixtureHome, join(temporary, 'mcp-cold.db')));
} finally {
  await rm(temporary, { recursive: true, force: true });
}

const report = {
  generatedAt: new Date().toISOString(),
  system: {
    platform: platform(),
    release: release(),
    architecture: arch(),
    cpu: cpus()[0]?.model ?? 'unknown',
    node: process.version,
  },
  databasePath: dbPath,
  databaseBytes: info.size,
  results,
};

function details(result) {
  const pieces = [];
  if (result.rows != null) pieces.push(`${result.rows} rows`);
  if (result.counters) {
    pieces.push(`${result.counters.filesOpened} files opened`);
    pieces.push(`${result.counters.bytesRead} bytes read`);
    pieces.push(`${result.counters.providerQueries} provider queries`);
    pieces.push(`${result.counters.recordsInspected} records inspected`);
    pieces.push(`${result.counters.skippedUnchanged} unchanged`);
  }
  return pieces.join('; ');
}

function markdown(value) {
  const lines = [
    '# RelayHistory native benchmark',
    '',
    `Generated: ${value.generatedAt}`,
    '',
    `- Platform: ${value.system.platform} ${value.system.release} (${value.system.architecture})`,
    `- CPU: ${value.system.cpu}`,
    `- Node: ${value.system.node}`,
    `- Database: \`${value.databasePath}\` (${value.databaseBytes.toLocaleString('en-US')} bytes)`,
    '',
    '| Operation | Time | Work |',
    '|---|---:|---|',
    ...value.results.map((result) => `| ${result.name} | ${result.ms.toFixed(2)} ms | ${details(result)} |`),
    '',
  ];
  return lines.join('\n');
}

function prettyTable(value) {
  const rows = value.results.map((result) => ({ name: result.name, ms: result.ms.toFixed(2) }));
  const nameWidth = Math.max('name'.length, ...rows.map((row) => row.name.length));
  const millisecondsWidth = Math.max('ms'.length, ...rows.map((row) => row.ms.length));
  return [
    `${'name'.padEnd(nameWidth)} | ${'ms'.padStart(millisecondsWidth)}`,
    `${'-'.repeat(nameWidth)}-+-${'-'.repeat(millisecondsWidth)}`,
    ...rows.map((row) => `${row.name.padEnd(nameWidth)} | ${row.ms.padStart(millisecondsWidth)}`),
    '',
  ].join('\n');
}

const rendered = outputPath && extname(outputPath).toLowerCase() === '.md'
  ? markdown(report)
  : `${JSON.stringify(report, null, 2)}\n`;

if (outputPath) {
  await writeFile(outputPath, rendered, 'utf8');
  if (!pretty) console.log(`RelayHistory benchmark written to ${outputPath}`);
}

if (pretty) {
  process.stdout.write(prettyTable(report));
} else if (!outputPath) {
  process.stdout.write(rendered);
}
