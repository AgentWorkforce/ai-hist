import { execFile } from 'node:child_process';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { arch, cpus, platform, release, tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  defaultDbPath, discoverSessions, getSessionEventsPage, listSessionCatalogPage,
} from '../sdk-ts/dist/index.js';

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

const dbPath = resolve(option('db') || process.env.AI_HIST_DB || defaultDbPath());
const outputOption = option('output');
const outputPath = outputOption ? resolve(invocationDirectory, outputOption) : null;
const discoveryLimit = Number(option('discovery-limit') || 100);
if (!Number.isInteger(discoveryLimit) || discoveryLimit < 1) {
  throw new Error('--discovery-limit must be a positive integer');
}

async function timed(name, operation) {
  const start = performance.now();
  const value = await operation();
  return { name, ms: Number((performance.now() - start).toFixed(2)), value };
}

async function mcpList() {
  const child = spawn(process.execPath, [join(repositoryRoot, 'sdk-ts/dist/mcp-server.js')], {
    cwd: repositoryRoot, env: { ...process.env, AI_HIST_DB: dbPath }, stdio: ['pipe', 'pipe', 'inherit'],
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
    const start = performance.now();
    await request('tools/call', { name: 'list_sessions', arguments: { limit: 20 } });
    return Number((performance.now() - start).toFixed(2));
  } finally {
    child.kill();
  }
}

const info = await stat(dbPath);
const results = [];
await listSessionCatalogPage({ dbPath, limit: 20 }); // initialize addon and warm OS cache
for (const limit of [20, 100]) {
  const measurement = await timed(`warm catalog ${limit}`, () => listSessionCatalogPage({ dbPath, limit }));
  results.push({ name: measurement.name, ms: measurement.ms, rows: measurement.value.sessions.length });
}

const firstPage = await listSessionCatalogPage({ dbPath, limit: 20 });
const candidate = firstPage.sessions.find((session) => session.discoveryState === 'full') ?? firstPage.sessions[0];
if (candidate) {
  const measurement = await timed('event page 200', () => getSessionEventsPage(candidate.sessionId, {
    dbPath, source: candidate.source, limit: 200,
  }));
  results.push({ name: measurement.name, ms: measurement.ms, rows: measurement.value.events.length });
}

const temporary = await mkdtemp(join(tmpdir(), 'relayhistory-benchmark-'));
try {
  const discoveryOptions = { dbPath: join(temporary, 'history.db'), sources: ['claude', 'codex'], limit: discoveryLimit };
  const first = await timed(`first shallow discovery ${discoveryLimit}`, () => discoverSessions(discoveryOptions));
  results.push({ name: first.name, ms: first.ms, rows: first.value.sessions.length, counters: first.value.counters });
  const unchanged = await timed(`unchanged shallow discovery ${discoveryLimit}`, () => discoverSessions(discoveryOptions));
  results.push({ name: unchanged.name, ms: unchanged.ms, rows: unchanged.value.sessions.length, counters: unchanged.value.counters });
} finally {
  await rm(temporary, { recursive: true, force: true });
}

const cli = await timed('CLI startup + catalog 20', () => execFileAsync(process.execPath, [
  join(repositoryRoot, 'sdk-ts/dist/cli.js'), 'sessions', 'list', '--db', dbPath, '--limit', '20', '--json',
], { maxBuffer: 10 * 1024 * 1024 }));
results.push({ name: cli.name, ms: cli.ms, rows: JSON.parse(cli.value.stdout).sessions.length });
results.push({ name: 'MCP list_sessions 20', ms: await mcpList(), rows: 20 });

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

const rendered = outputPath && extname(outputPath).toLowerCase() === '.md'
  ? markdown(report)
  : `${JSON.stringify(report, null, 2)}\n`;

if (outputPath) {
  await writeFile(outputPath, rendered, 'utf8');
  console.log(`RelayHistory benchmark written to ${outputPath}`);
} else {
  process.stdout.write(rendered);
}
