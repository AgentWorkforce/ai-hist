import { writeFile } from 'node:fs/promises';
import { arch, cpus, platform, release } from 'node:os';
import { dirname, extname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import {
  defaultDbPath, hydrateSession, listSessionCatalogPage, nativeBuildProfile,
} from '../sdk-ts/dist/index.js';
import { durationSummary, selectHydrationSessions } from './hydration-benchmark-lib.mjs';

const invocationDirectory = process.cwd();
const supportedSources = new Set(['claude', 'codex', 'cursor', 'grok', 'opencode']);

function option(name, fallback) {
  const prefix = `--${name}=`;
  const inline = process.argv.find((argument) => argument.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  if (index >= 0) return process.argv[index + 1];
  return process.env[`npm_config_${name.replaceAll('-', '_')}`] ?? fallback;
}

function options(name) {
  const values = [];
  for (let index = 2; index < process.argv.length; index++) {
    if (process.argv[index] === `--${name}` && process.argv[index + 1]) values.push(process.argv[++index]);
    else if (process.argv[index].startsWith(`--${name}=`)) values.push(process.argv[index].slice(name.length + 3));
  }
  return values;
}

function flag(name) {
  if (process.argv.includes(`--${name}`)) return true;
  return ['1', 'true', 'yes'].includes(process.env[`npm_config_${name.replaceAll('-', '_')}`]?.toLowerCase());
}

function positiveInteger(name, fallback) {
  const value = Number(option(name, fallback));
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}

async function catalogCandidates(dbPath, sources, scanLimit) {
  const sessions = [];
  let after;
  while (sessions.length < scanLimit) {
    const page = await listSessionCatalogPage({
      dbPath,
      sources: sources.length ? sources : undefined,
      limit: Math.min(100, scanLimit - sessions.length),
      after,
    });
    sessions.push(...page.sessions);
    if (!page.nextCursor) break;
    after = page.nextCursor;
  }
  return sessions;
}

async function measureHydration(session, dbPath, iterations, includeRelated) {
  const invoke = async () => {
    const started = performance.now();
    const value = await hydrateSession({
      dbPath,
      source: session.source,
      sessionId: session.sessionId,
      includeRelated,
    });
    return { ms: performance.now() - started, value };
  };
  try {
    const first = await invoke();
    const repeats = [];
    for (let index = 0; index < iterations; index++) repeats.push(await invoke());
    const repeatStatuses = Object.fromEntries(
      ['hydrated', 'updated', 'unchanged'].map((status) => [
        status, repeats.filter(({ value }) => value.status === status).length,
      ]),
    );
    const diagnostic = first.value.diagnostics[0];
    return {
      source: session.source,
      sessionId: session.sessionId,
      priorDiscoveryState: session.discoveryState,
      first: {
        status: first.value.status,
        ms: Number(first.ms.toFixed(2)),
        evidence: first.value.evidence,
        relatedSessions: first.value.relatedSessionIds.length,
        sourceBytes: diagnostic?.sourceBytes ?? null,
        recordsParsed: diagnostic?.recordsParsed ?? null,
      },
      unchanged: durationSummary(
        repeats.filter(({ value }) => value.status === 'unchanged').map(({ ms }) => ms),
      ),
      repeatStatuses,
    };
  } catch (error) {
    return {
      source: session.source,
      sessionId: session.sessionId,
      priorDiscoveryState: session.discoveryState,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function markdown(report) {
  const lines = [
    '# RelayHistory hydration benchmark',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    `- Database: \`${report.databasePath}\``,
    `- Selected sessions: ${report.selectedCount}`,
    `- Unchanged iterations per session: ${report.iterations}`,
    `- Include related sessions: ${report.includeRelated}`,
    '',
    '| Session | Previous state | First status | First | Unchanged p50 | Unchanged p95 | Evidence |',
    '|---|---|---|---:|---:|---:|---|',
  ];
  for (const result of report.results) {
    const id = `${result.source}:${result.sessionId}`;
    if (result.error) {
      lines.push(`| ${id} | ${result.priorDiscoveryState} | error | — | — | — | ${result.error.replaceAll('|', '\\|')} |`);
      continue;
    }
    const evidence = result.first.evidence;
    const p50 = result.unchanged ? `${result.unchanged.p50Ms.toFixed(2)} ms` : '—';
    const p95 = result.unchanged ? `${result.unchanged.p95Ms.toFixed(2)} ms` : '—';
    lines.push(`| ${id} | ${result.priorDiscoveryState} | ${result.first.status} | ${result.first.ms.toFixed(2)} ms | ${p50} | ${p95} | ${evidence.prompts} prompts; ${evidence.events} events; ${evidence.toolCalls} tools |`);
  }
  lines.push('');
  return lines.join('\n');
}

function prettyTable(report) {
  const rows = report.results.map((result) => ({
    session: `${result.source}:${result.sessionId}`,
    status: result.error ? 'error' : result.first.status,
    first: result.error ? '—' : result.first.ms.toFixed(2),
    p50: result.error || !result.unchanged ? '—' : result.unchanged.p50Ms.toFixed(2),
    p95: result.error || !result.unchanged ? '—' : result.unchanged.p95Ms.toFixed(2),
  }));
  const widths = Object.fromEntries(['session', 'status', 'first', 'p50', 'p95'].map((key) => [
    key, Math.max(key.length, ...rows.map((row) => row[key].length)),
  ]));
  const row = (value) => `${value.session.padEnd(widths.session)} | ${value.status.padEnd(widths.status)} | ${value.first.padStart(widths.first)} | ${value.p50.padStart(widths.p50)} | ${value.p95.padStart(widths.p95)}`;
  return `${row({ session: 'session', status: 'status', first: 'first', p50: 'p50', p95: 'p95' })}\n${'-'.repeat(widths.session + widths.status + widths.first + widths.p50 + widths.p95 + 12)}\n${rows.map(row).join('\n')}\n`;
}

async function main() {
  const buildProfile = await nativeBuildProfile();
  if (buildProfile !== 'release') {
    throw new Error(`Hydration benchmarks require a release native addon, not ${buildProfile}. Run: npm run build --prefix crates/ai-hist-napi`);
  }
  const dbPath = resolve(option('db', process.env.AI_HIST_DB || defaultDbPath()));
  const count = positiveInteger('count', 5);
  const iterations = positiveInteger('iterations', 5);
  const sourceFilters = options('source');
  for (const source of sourceFilters) {
    if (!supportedSources.has(source)) throw new Error(`unsupported --source ${source}`);
  }
  const candidates = await catalogCandidates(dbPath, sourceFilters, Math.max(100, count * 20));
  const selected = selectHydrationSessions(candidates, count);
  if (selected.length === 0) {
    throw new Error('No hydratable local catalog sessions with provider locators were found. Run sessions discover first.');
  }
  const includeRelated = flag('include-related');
  const results = [];
  for (const session of selected) {
    results.push(await measureHydration(session, dbPath, iterations, includeRelated));
  }
  const report = {
    generatedAt: new Date().toISOString(),
    system: {
      platform: platform(), release: release(), architecture: arch(),
      cpu: cpus()[0]?.model ?? 'unknown', node: process.version,
    },
    databasePath: dbPath,
    selectedCount: selected.length,
    requestedCount: count,
    iterations,
    includeRelated,
    results,
  };
  const output = option('output');
  if (output) {
    const outputPath = resolve(invocationDirectory, output);
    const rendered = extname(outputPath).toLowerCase() === '.md'
      ? markdown(report)
      : `${JSON.stringify(report, null, 2)}\n`;
    await writeFile(outputPath, rendered, 'utf8');
    if (!flag('pretty')) console.log(`RelayHistory hydration benchmark written to ${outputPath}`);
  }
  if (flag('pretty')) process.stdout.write(prettyTable(report));
  else if (!output) process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
