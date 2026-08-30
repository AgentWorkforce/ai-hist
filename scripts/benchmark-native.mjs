import { execFile, spawn } from 'node:child_process';
import { appendFile, mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises';
import { arch, cpus, platform, release, tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
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
    return { name: measurement.name, ms: measurement.ms, rows: 20 };
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

async function withHome(home, operation) {
  const previousHome = process.env.HOME;
  const previousProfile = process.env.USERPROFILE;
  process.env.HOME = home;
  process.env.USERPROFILE = home;
  try {
    return await operation();
  } finally {
    if (previousHome === undefined) delete process.env.HOME;
    else process.env.HOME = previousHome;
    if (previousProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = previousProfile;
  }
}

async function discoverFixture(home, benchmarkDbPath, limit) {
  return withHome(home, () => discoverSessions({
    dbPath: benchmarkDbPath, sources: ['claude'], limit,
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

const info = await stat(dbPath);
const results = [];
const temporary = await mkdtemp(join(tmpdir(), 'relayhistory-benchmark-'));
try {
  const fixtureHome = join(temporary, 'home');
  await createDiscoveryFixture(fixtureHome, Math.max(...discoveryLimits));

  const coldResults = [];
  const unchangedResults = [];
  const changedResults = [];
  for (const limit of discoveryLimits) {
    const coldDb = join(temporary, `cold-${limit}.db`);
    const cold = await timed(`cold shallow discovery ${limit}`, () => discoverFixture(fixtureHome, coldDb, limit));
    coldResults.push({ name: cold.name, ms: cold.ms, rows: cold.value.sessions.length, counters: cold.value.counters });

    const unchangedDb = join(temporary, `unchanged-${limit}.db`);
    await discoverFixture(fixtureHome, unchangedDb, limit);
    const unchanged = await timed(`unchanged shallow discovery ${limit}`, () => discoverFixture(fixtureHome, unchangedDb, limit));
    unchangedResults.push({ name: unchanged.name, ms: unchanged.ms, rows: unchanged.value.sessions.length, counters: unchanged.value.counters });

    const changedDb = join(temporary, `changed-${limit}.db`);
    const seeded = await discoverFixture(fixtureHome, changedDb, limit);
    const originals = await changeDiscoveredSessions(seeded.sessions);
    try {
      const changed = await timed(`cold->changed shallow discovery ${limit}`, () => discoverFixture(fixtureHome, changedDb, limit));
      changedResults.push({ name: changed.name, ms: changed.ms, rows: changed.value.sessions.length, counters: changed.value.counters });
    } finally {
      await restoreFixtureSessions(originals);
    }
  }
  results.push(...coldResults, ...unchangedResults, ...changedResults);

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
  results.push({ name: cli.name, ms: cli.ms, rows: JSON.parse(cli.value.stdout).sessions.length });

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
