#!/usr/bin/env node

import { readFile } from 'node:fs/promises';

import {
  discoverSessions, getSession, getSessionEventsPage, listSessionCatalogPage,
  recent, search, stats, sync, type CatalogCursor, type SessionScope,
} from './index.js';

type Parsed = { positional: string[]; flags: Map<string, Array<string | true>> };

type PackageMetadata = { version?: string };

const BOOLEAN_FLAGS = new Set(['all', 'fts', 'json', 'local', 'no-warning', 'remote', 'version']);
const VALUE_FLAGS = new Set([
  'after', 'after-ms', 'after-session-id', 'after-source', 'before-ms', 'db', 'limit',
  'project', 'source', 'tag',
]);
const KNOWN_FLAGS = new Set([...BOOLEAN_FLAGS, ...VALUE_FLAGS]);

function versionTriple(value: string): [number, number, number] | null {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$/.exec(value);
  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : null;
}

function newerVersion(current: string, latest: string): boolean {
  const left = versionTriple(current);
  const right = versionTriple(latest);
  if (!left || !right) return false;
  for (let index = 0; index < left.length; index++) {
    if (right[index] !== left[index]) return right[index] > left[index];
  }
  return false;
}

async function packageVersion(): Promise<string> {
  const contents = await readFile(new URL('../package.json', import.meta.url), 'utf8');
  return (JSON.parse(contents) as PackageMetadata).version ?? 'unknown';
}

async function maybePrintUpdateNotice(current: string, args: string[]): Promise<void> {
  const optOut = process.env.RELAYHISTORY_NO_UPDATE_CHECK;
  if (!process.stderr.isTTY || args.includes('--no-warning') || (optOut && optOut !== '0')) return;
  try {
    const response = await fetch('https://registry.npmjs.org/ai-hist/latest', {
      signal: AbortSignal.timeout(3_000),
      headers: { accept: 'application/json' },
    });
    if (!response.ok) return;
    const latest = (await response.json()) as PackageMetadata;
    if (!latest.version || !newerVersion(current, latest.version)) return;
    process.stderr.write(
      `\nA new version of ai-hist is available: ${current} -> ${latest.version}\n` +
      'Update with:\n  npm install --global ai-hist@latest\n' +
      '(pass --no-warning or set RELAYHISTORY_NO_UPDATE_CHECK=1 to hide this notice)\n',
    );
  } catch {
    // Version checks are best-effort; --version stays useful while offline.
  }
}

function parse(argv: string[]): Parsed {
  const positional: string[] = [];
  const flags = new Map<string, Array<string | true>>();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-V') {
      flags.set('version', [...(flags.get('version') ?? []), true]);
      continue;
    }
    if (!arg.startsWith('--')) { positional.push(arg); continue; }
    const [name, inline] = arg.slice(2).split('=', 2);
    if (!KNOWN_FLAGS.has(name)) usage(`unknown option '--${name}'`);
    if (inline !== undefined && BOOLEAN_FLAGS.has(name)) usage(`--${name} does not take a value`);
    if (inline !== undefined) {
      if (inline === '') usage(`--${name} requires a value`);
      flags.set(name, [...(flags.get(name) ?? []), inline]);
      continue;
    }
    if (BOOLEAN_FLAGS.has(name)) {
      const next = argv[i + 1];
      if (next === 'true' || next === 'false') usage(`--${name} does not take a value`);
      flags.set(name, [...(flags.get(name) ?? []), true]);
      continue;
    }
    const next = argv[i + 1];
    if (next && !next.startsWith('-')) { flags.set(name, [...(flags.get(name) ?? []), next]); i++; }
    else usage(`--${name} requires a value`);
  }
  return { positional, flags };
}

function validateFlags(args: Parsed, command: string, allowed: readonly string[]): void {
  const permitted = new Set([...allowed, 'no-warning']);
  for (const name of args.flags.keys()) {
    if (!permitted.has(name)) usage(`${command} does not accept --${name}`);
  }
}

function textFlag(args: Parsed, name: string): string | undefined {
  const value = args.flags.get(name)?.at(-1);
  return typeof value === 'string' ? value : undefined;
}

function textFlags(args: Parsed, name: string): string[] {
  return (args.flags.get(name) ?? []).filter((value): value is string => typeof value === 'string');
}

function numberFlag(args: Parsed, name: string): number | undefined {
  const value = textFlag(args, name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`--${name} must be a number`);
  return parsed;
}

function scopeFlag(args: Parsed): SessionScope {
  const selected = (['local', 'remote', 'all'] as const).filter((scope) => args.flags.has(scope));
  for (const scope of selected) {
    if (args.flags.get(scope)?.some((value) => value !== true)) {
      usage(`--${scope} does not take a value`);
    }
  }
  if (selected.length > 1) usage('--local, --remote, and --all are mutually exclusive');
  return selected[0] ?? 'local';
}

function rejectScopeFlag(args: Parsed, command: string): void {
  if (['local', 'remote', 'all'].some((scope) => args.flags.has(scope))) {
    usage(`${command} addresses a session by identity and does not accept --local, --remote, or --all`);
  }
}

function rejectSurplusPositionals(values: string[], command: string): void {
  if (values.length > 0) usage(`${command} does not accept positional argument '${values[0]}'`);
}

function common(args: Parsed) {
  return {
    dbPath: textFlag(args, 'db'),
    scope: scopeFlag(args),
    source: textFlag(args, 'source') as never,
    project: textFlag(args, 'project'),
    tag: textFlag(args, 'tag'),
    limit: numberFlag(args, 'limit'),
    beforeMs: numberFlag(args, 'before-ms'),
  };
}

function snakeCase(key: string): string {
  return key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

function wireValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(wireValue);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [snakeCase(key), wireValue(item)]));
}

function humanLine(value: unknown): string {
  if (!value || typeof value !== 'object') return String(value);
  const row = value as Record<string, unknown>;
  const locations = Array.isArray(row.locations) ? `[${row.locations.join(',')}]` : '';
  return [row.timestampMs ?? row.lastActivityMs ?? '', row.source ?? '', locations, row.sessionId ?? '', row.project ?? row.cwd ?? '', row.prompt ?? row.firstPrompt ?? '']
    .filter((item) => item !== '' && item != null)
    .join('  ');
}

function output(value: unknown, json: boolean): void {
  if (json) {
    process.stdout.write(`${JSON.stringify(wireValue(value))}\n`);
  } else if (Array.isArray(value)) {
    process.stdout.write(value.length ? `${value.map(humanLine).join('\n')}\n` : 'No results.\n');
  } else if (typeof value === 'object' && value !== null) {
    const record = value as Record<string, unknown>;
    if (Array.isArray(record.sessions)) {
      process.stdout.write(record.sessions.length ? `${record.sessions.map(humanLine).join('\n')}\n` : 'No sessions in the catalog.\n');
      if (record.nextCursor) process.stdout.write(`more available: --after '${JSON.stringify(record.nextCursor)}'\n`);
    } else {
      process.stdout.write(`${Object.entries(record).map(([key, item]) => `${key}: ${typeof item === 'object' ? JSON.stringify(item) : String(item)}`).join('\n')}\n`);
    }
  } else {
    process.stdout.write(`${String(value)}\n`);
  }
}

function usage(message?: string): never {
  if (message) process.stderr.write(`ai-hist: ${message}\n\n`);
  process.stderr.write(`Usage:
  ai-hist sessions list [--local | --remote | --all] [--source SOURCE]... [--limit N] [--before-ms MS] [--after JSON | --after-source SOURCE --after-session-id ID [--after-ms MS]] [--json]
  ai-hist sessions discover [--local | --remote | --all] [--source SOURCE] [--limit N] [--json]
  ai-hist search QUERY... [--local | --remote | --all] [--source SOURCE] [--project PATH] [--limit N] [--json]
  ai-hist recent [N] [--local | --remote | --all] [--source SOURCE] [--project PATH] [--json]
  ai-hist session SESSION_ID [--source SOURCE] [--json]
  ai-hist events SESSION_ID [--source SOURCE] [--limit N] [--after JSON] [--json]
  ai-hist stats [--local | --remote | --all] [--json]
  ai-hist sync [--local | --remote | --all] [--db PATH] [--json]
`);
  process.exit(2);
}

function cursorFlag<T>(args: Parsed): T | undefined {
  const raw = textFlag(args, 'after');
  return raw ? JSON.parse(raw) as T : undefined;
}

function catalogCursorFlag(args: Parsed): CatalogCursor | undefined {
  const encoded = cursorFlag<CatalogCursor>(args);
  if (encoded) return encoded;
  const source = textFlag(args, 'after-source');
  const sessionId = textFlag(args, 'after-session-id');
  if (!source && !sessionId) return undefined;
  if (!source || !sessionId) throw new Error('--after-source and --after-session-id must be used together');
  return { lastActivityMs: numberFlag(args, 'after-ms') ?? null, source, sessionId };
}

function outputDiscovery(value: Awaited<ReturnType<typeof discoverSessions>>, json: boolean): void {
  if (!json) {
    for (const session of value.sessions) process.stdout.write(`${humanLine(session)}\n`);
    process.stdout.write(
      `${value.sessions.length} session(s): ${value.discovered} discovered, ${value.skippedUnchanged} unchanged ` +
      `(${value.counters.filesOpened} file(s) opened, ${value.counters.shallowReads} shallow read(s)); ` +
      `requested scope: ${value.scope}, connector locations run: ${value.locationsRun.length > 0 ? value.locationsRun.join(', ') : 'none'}\n`,
    );
    return;
  }
  for (const session of value.sessions) output({ type: 'session', ...session }, true);
  for (const diagnostic of value.diagnostics) output({ type: 'diagnostic', ...diagnostic }, true);
  const { sessions: _sessions, diagnostics: _diagnostics, ...summary } = value;
  const providers = Object.fromEntries(summary.providers.map(({ source, ...provider }) => [source, provider]));
  output({ type: 'summary', ...summary, providers }, true);
}

async function main(): Promise<void> {
  const rawArgs = process.argv.slice(2);
  const versionArgs = rawArgs.filter((arg) => arg !== '--no-warning');
  if (versionArgs.length === 1 && (versionArgs[0] === '--version' || versionArgs[0] === '-V')) {
    const version = await packageVersion();
    process.stdout.write(`ai-hist ${version}\n`);
    await maybePrintUpdateNotice(version, rawArgs);
    return;
  }
  const args = parse(rawArgs);
  const [command, subcommand, ...rest] = args.positional;
  const json = args.flags.has('json');

  if (command === 'sessions' && subcommand === 'list') {
    validateFlags(args, 'sessions list', [
      'after', 'after-ms', 'after-session-id', 'after-source', 'all', 'before-ms', 'db',
      'json', 'limit', 'local', 'remote', 'source',
    ]);
    rejectSurplusPositionals(rest, 'sessions list');
    const sources = textFlags(args, 'source');
    output(await listSessionCatalogPage({
      dbPath: textFlag(args, 'db'), scope: scopeFlag(args), sources: sources.length ? sources as never : undefined,
      limit: numberFlag(args, 'limit'), beforeMs: numberFlag(args, 'before-ms'),
      after: catalogCursorFlag(args),
    }), json);
    return;
  }
  if (command === 'sessions' && subcommand === 'discover') {
    validateFlags(args, 'sessions discover', ['all', 'db', 'json', 'limit', 'local', 'remote', 'source']);
    rejectSurplusPositionals(rest, 'sessions discover');
    const sources = textFlags(args, 'source');
    outputDiscovery(await discoverSessions({
      dbPath: textFlag(args, 'db'), scope: scopeFlag(args), sources: sources.length ? sources as never : undefined,
      limit: numberFlag(args, 'limit'),
    }), json);
    return;
  }
  if (command === 'search') {
    validateFlags(args, 'search', [
      'all', 'before-ms', 'db', 'fts', 'json', 'limit', 'local', 'project', 'remote', 'source', 'tag',
    ]);
    if (!subcommand) usage();
    output(await search([subcommand, ...rest].join(' '), { ...common(args), rawFts: args.flags.has('fts') }), json);
    return;
  }
  if (command === 'recent') {
    validateFlags(args, 'recent', [
      'all', 'before-ms', 'db', 'json', 'limit', 'local', 'project', 'remote', 'source', 'tag',
    ]);
    rejectSurplusPositionals(rest, 'recent');
    const fallback = subcommand ? Number(subcommand) : undefined;
    if (fallback !== undefined && !Number.isFinite(fallback)) usage(`recent count must be a number (got '${subcommand}')`);
    output(await recent({ ...common(args), limit: numberFlag(args, 'limit') ?? fallback }), json);
    return;
  }
  if (command === 'session') {
    validateFlags(args, 'session', ['all', 'db', 'json', 'local', 'remote', 'source', 'tag']);
    if (!subcommand) usage();
    rejectSurplusPositionals(rest, 'session');
    rejectScopeFlag(args, 'session');
    output(await getSession(subcommand, { dbPath: textFlag(args, 'db'), source: textFlag(args, 'source') as never, tag: textFlag(args, 'tag') }), json);
    return;
  }
  if (command === 'events') {
    validateFlags(args, 'events', ['after', 'all', 'db', 'json', 'limit', 'local', 'remote', 'source']);
    if (!subcommand) usage();
    rejectSurplusPositionals(rest, 'events');
    rejectScopeFlag(args, 'events');
    output(await getSessionEventsPage(subcommand, {
      dbPath: textFlag(args, 'db'), source: textFlag(args, 'source') as never,
      limit: numberFlag(args, 'limit'), after: cursorFlag(args),
    }), json);
    return;
  }
  if (command === 'stats') {
    validateFlags(args, 'stats', ['all', 'db', 'json', 'local', 'remote', 'tag']);
    rejectSurplusPositionals([subcommand, ...rest].filter((value): value is string => value !== undefined), 'stats');
    output(await stats({ dbPath: textFlag(args, 'db'), scope: scopeFlag(args), tag: textFlag(args, 'tag') }), json);
    return;
  }
  if (command === 'sync') {
    validateFlags(args, 'sync', ['all', 'db', 'json', 'local', 'remote']);
    rejectSurplusPositionals([subcommand, ...rest].filter((value): value is string => value !== undefined), 'sync');
    output(await sync({ dbPath: textFlag(args, 'db'), scope: scopeFlag(args) }), json);
    return;
  }
  usage();
}

main().catch((error: unknown) => {
  const value = error as { code?: string; message?: string };
  process.stderr.write(`ai-hist: ${value.code ? `${value.code}: ` : ''}${value.message ?? String(error)}\n`);
  process.exitCode = 1;
});
