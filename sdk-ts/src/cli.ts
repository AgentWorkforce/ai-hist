#!/usr/bin/env node

import { readFile } from 'node:fs/promises';

import {
  discoverSessions, getSession, getSessionEventsPage, listSessionCatalogPage,
  recent, search, stats, sync, type CatalogCursor,
} from './index.js';

type Parsed = { positional: string[]; flags: Map<string, string | true> };

type PackageMetadata = { version?: string };

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
  const flags = new Map<string, string | true>();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) { positional.push(arg); continue; }
    const [name, inline] = arg.slice(2).split('=', 2);
    if (inline !== undefined) { flags.set(name, inline); continue; }
    const next = argv[i + 1];
    if (next && !next.startsWith('-')) { flags.set(name, next); i++; }
    else flags.set(name, true);
  }
  return { positional, flags };
}

function textFlag(args: Parsed, name: string): string | undefined {
  const value = args.flags.get(name);
  return typeof value === 'string' ? value : undefined;
}

function numberFlag(args: Parsed, name: string): number | undefined {
  const value = textFlag(args, name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`--${name} must be a number`);
  return parsed;
}

function common(args: Parsed) {
  return {
    dbPath: textFlag(args, 'db'),
    source: textFlag(args, 'source') as never,
    project: textFlag(args, 'project'),
    tag: textFlag(args, 'tag'),
    limit: numberFlag(args, 'limit'),
    beforeMs: numberFlag(args, 'before-ms'),
  };
}

function output(value: unknown, json: boolean): void {
  if (json || typeof value !== 'string') process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
  else process.stdout.write(`${value}\n`);
}

function usage(): never {
  process.stderr.write(`Usage:
  ai-hist sessions list [--source SOURCE] [--limit N] [--before-ms MS] [--after JSON] [--json]
  ai-hist sessions discover [--source SOURCE] [--limit N] [--json]
  ai-hist search QUERY... [--source SOURCE] [--project PATH] [--limit N] [--json]
  ai-hist recent [N] [--source SOURCE] [--project PATH] [--json]
  ai-hist session SESSION_ID [--source SOURCE] [--json]
  ai-hist events SESSION_ID [--source SOURCE] [--limit N] [--after JSON] [--json]
  ai-hist stats [--json]
  ai-hist sync [--db PATH] [--json]
`);
  process.exit(2);
}

function cursorFlag<T>(args: Parsed): T | undefined {
  const raw = textFlag(args, 'after');
  return raw ? JSON.parse(raw) as T : undefined;
}

async function main(): Promise<void> {
  const rawArgs = process.argv.slice(2);
  if (rawArgs.includes('--version') || rawArgs.includes('-V')) {
    const version = await packageVersion();
    process.stdout.write(`ai-hist ${version}\n`);
    await maybePrintUpdateNotice(version, rawArgs);
    return;
  }
  const args = parse(rawArgs);
  const [command, subcommand, ...rest] = args.positional;
  const json = args.flags.has('json');

  if (command === 'sessions' && subcommand === 'list') {
    const source = textFlag(args, 'source');
    output(await listSessionCatalogPage({
      dbPath: textFlag(args, 'db'), sources: source ? [source as never] : undefined,
      limit: numberFlag(args, 'limit'), beforeMs: numberFlag(args, 'before-ms'),
      after: cursorFlag<CatalogCursor>(args),
    }), true);
    return;
  }
  if (command === 'sessions' && subcommand === 'discover') {
    const source = textFlag(args, 'source');
    output(await discoverSessions({
      dbPath: textFlag(args, 'db'), sources: source ? [source as never] : undefined,
      limit: numberFlag(args, 'limit'),
    }), true);
    return;
  }
  if (command === 'search') {
    if (!subcommand) usage();
    output(await search([subcommand, ...rest].join(' '), { ...common(args), rawFts: args.flags.has('fts') }), json);
    return;
  }
  if (command === 'recent') {
    const fallback = subcommand ? Number(subcommand) : undefined;
    output(await recent({ ...common(args), limit: numberFlag(args, 'limit') ?? fallback }), json);
    return;
  }
  if (command === 'session') {
    if (!subcommand) usage();
    output(await getSession(subcommand, { dbPath: textFlag(args, 'db'), source: textFlag(args, 'source') as never, tag: textFlag(args, 'tag') }), json);
    return;
  }
  if (command === 'events') {
    if (!subcommand) usage();
    output(await getSessionEventsPage(subcommand, {
      dbPath: textFlag(args, 'db'), source: textFlag(args, 'source') as never,
      limit: numberFlag(args, 'limit'), after: cursorFlag(args),
    }), true);
    return;
  }
  if (command === 'stats') { output(await stats({ dbPath: textFlag(args, 'db'), tag: textFlag(args, 'tag') }), true); return; }
  if (command === 'sync') { output(await sync({ dbPath: textFlag(args, 'db') }), true); return; }
  usage();
}

main().catch((error: unknown) => {
  const value = error as { code?: string; message?: string };
  process.stderr.write(`ai-hist: ${value.code ? `${value.code}: ` : ''}${value.message ?? String(error)}\n`);
  process.exitCode = 1;
});
