import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const run = promisify(execFile);
const cli = join(dirname(fileURLToPath(import.meta.url)), 'cli.js');

function isUsageFailure(error: unknown, message: string): boolean {
  return typeof error === 'object' && error !== null
    && 'code' in error && error.code === 2
    && 'stderr' in error && String(error.stderr).includes(message)
    && String(error.stderr).includes('Usage:');
}

test('sessions discover preserves repeated sources and emits JSONL', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-'));
  const home = join(root, 'home');
  const claude = join(home, '.claude', 'projects', 'project');
  const codex = join(home, '.codex', 'sessions', '2026', '08', '30');
  await mkdir(claude, { recursive: true });
  await mkdir(codex, { recursive: true });
  await writeFile(join(home, '.claude', 'history.jsonl'), `${JSON.stringify({
    display: 'indexed prompt', sessionId: 'claude-1', project: '/work/claude', timestamp: 1,
  })}\n`);
  await writeFile(join(claude, 'claude-1.jsonl'), `${JSON.stringify({
    sessionId: 'claude-1', cwd: '/work/claude', type: 'user',
    message: { role: 'user', content: 'claude prompt' }, timestamp: '2026-08-30T10:00:00.000Z',
  })}\n`);
  await writeFile(join(codex, 'rollout-codex-1.jsonl'), `${JSON.stringify({
    timestamp: '2026-08-30T11:00:00.000Z', type: 'session_meta',
    payload: { id: 'codex-1', cwd: '/work/codex' },
  })}\n${JSON.stringify({
    timestamp: '2026-08-30T11:00:01.000Z', type: 'event_msg',
    payload: { type: 'user_message', message: 'codex prompt' },
  })}\n`);

  try {
    const { stdout } = await run(process.execPath, [
      cli, 'sessions', 'discover', '--source', 'claude', '--source', 'codex',
      '--db', join(root, 'history.db'), '--json', '--no-warning',
    ], { env: { ...process.env, HOME: home, USERPROFILE: home } });
    const lines = stdout.trim().split('\n').map((line) => JSON.parse(line) as Record<string, unknown>);
    const sessions = lines.filter((line) => line.type === 'session');
    assert.deepEqual(sessions.map((line) => line.source).sort(), ['claude', 'codex']);
    assert.ok(sessions.every((line) => JSON.stringify(line.locations) === '["local"]'));
    assert.equal(lines.at(-1)?.type, 'summary');
    assert.equal(lines.at(-1)?.contract_version, 2);
    assert.equal(lines.at(-1)?.scope, 'local');
    assert.equal('sessions' in (lines.at(-1) ?? {}), false);

    const human = await run(process.execPath, [
      cli, 'sessions', 'discover', '--all', '--source', 'codex',
      '--db', join(root, 'history.db'), '--no-warning',
    ], { env: { ...process.env, HOME: home, USERPROFILE: home } });
    assert.match(human.stdout, /requested scope: all, connector locations run: local/);

    await run(process.execPath, [
      cli, 'sync', '--db', join(root, 'history.db'), '--json', '--no-warning',
    ], { env: { ...process.env, HOME: home, USERPROFILE: home } });
    const searched = await run(process.execPath, [
      cli, 'search', 'indexed prompt', '--db', join(root, 'history.db'), '--json', '--no-warning',
    ], { env: { ...process.env, HOME: home, USERPROFILE: home } });
    assert.deepEqual((JSON.parse(searched.stdout) as Array<Record<string, unknown>>)[0]?.locations, ['local']);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('sessions list keeps human and JSON output contracts distinct', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-empty-'));
  const db = join(root, 'missing.db');
  try {
    const human = await run(process.execPath, [cli, 'sessions', 'list', '--db', db, '--no-warning']);
    assert.match(human.stdout, /No sessions in the catalog/);
    const json = await run(process.execPath, [cli, 'sessions', 'list', '--db', db, '--json', '--no-warning']);
    assert.deepEqual(JSON.parse(json.stdout), { contract_version: 2, scope: 'local', sessions: [], next_cursor: null });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('scope flags are boolean, default to local, and are mutually exclusive', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-scope-'));
  const db = join(root, 'missing.db');
  try {
    const implicit = await run(process.execPath, [cli, 'sessions', 'list', '--db', db, '--json', '--no-warning']);
    const explicit = await run(process.execPath, [cli, '--json', 'sessions', 'list', '--local', '--db', db, '--no-warning']);
    assert.deepEqual(JSON.parse(explicit.stdout), JSON.parse(implicit.stdout));
    const remote = await run(process.execPath, [cli, 'sessions', 'list', '--remote', '--db', db, '--json', '--no-warning']);
    assert.equal(JSON.parse(remote.stdout).scope, 'remote');

    await assert.rejects(
      run(process.execPath, [cli, 'sessions', 'list', '--local', '--remote', '--db', db, '--no-warning']),
      (error: unknown) => typeof error === 'object' && error !== null
        && 'stderr' in error
        && String(error.stderr).includes('--local, --remote, and --all are mutually exclusive'),
    );
    await assert.rejects(
      run(process.execPath, [cli, 'sessions', 'list', '--remote=false', '--db', db, '--no-warning']),
      (error: unknown) => isUsageFailure(error, '--remote does not take a value'),
    );
    await assert.rejects(
      run(process.execPath, [cli, 'sessions', 'list', '--remote', 'false', '--db', db, '--no-warning']),
      (error: unknown) => isUsageFailure(error, '--remote does not take a value'),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('commands reject surplus positional arguments with usage exit 2', async () => {
  for (const args of [
    ['sessions', 'list', 'extra'],
    ['sessions', 'discover', 'extra'],
    ['recent', '1', 'extra'],
    ['session', 'session-id', 'extra'],
    ['events', 'session-id', 'extra'],
    ['stats', 'extra'],
    ['sync', 'extra'],
  ]) {
    await assert.rejects(
      run(process.execPath, [cli, ...args, '--no-warning']),
      (error: unknown) => isUsageFailure(error, 'does not accept positional argument'),
      args.join(' '),
    );
  }
});

test('identity commands reject location scope instead of silently ignoring it', async () => {
  for (const command of ['session', 'events']) {
    await assert.rejects(
      run(process.execPath, [cli, command, 'session-id', '--remote', '--no-warning']),
      (error: unknown) => typeof error === 'object' && error !== null
        && 'stderr' in error
        && String(error.stderr).includes('does not accept --local, --remote, or --all'),
    );
  }
});

test('unknown, missing-value, and command-inapplicable flags fail with usage exit 2', async () => {
  for (const { args, message } of [
    { args: ['sessions', 'list', '--remtoe'], message: "unknown option '--remtoe'" },
    { args: ['sessions', 'list', '--db'], message: '--db requires a value' },
    { args: ['sessions', 'list', '--source'], message: '--source requires a value' },
    { args: ['sessions', 'list', '--project', '/work'], message: 'sessions list does not accept --project' },
    { args: ['sessions', 'list', '--version'], message: 'sessions list does not accept --version' },
    { args: ['sessions', 'list', '-V'], message: 'sessions list does not accept --version' },
    { args: ['stats', '--source', 'codex'], message: 'stats does not accept --source' },
  ]) {
    await assert.rejects(
      run(process.execPath, [cli, ...args, '--no-warning']),
      (error: unknown) => isUsageFailure(error, message),
      args.join(' '),
    );
  }
});

test('search preserves a multi-word positional query', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-search-'));
  try {
    const result = await run(process.execPath, [
      cli, 'search', 'two', 'word query', '--db', join(root, 'missing.db'), '--json', '--no-warning',
    ]);
    assert.deepEqual(JSON.parse(result.stdout), []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
