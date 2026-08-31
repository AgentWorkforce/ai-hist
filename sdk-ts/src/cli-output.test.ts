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

test('sessions discover preserves repeated sources and emits JSONL', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-'));
  const home = join(root, 'home');
  const claude = join(home, '.claude', 'projects', 'project');
  const codex = join(home, '.codex', 'sessions', '2026', '08', '30');
  await mkdir(claude, { recursive: true });
  await mkdir(codex, { recursive: true });
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
      (error: unknown) => typeof error === 'object' && error !== null
        && 'stderr' in error
        && String(error.stderr).includes('--remote does not take a value'),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
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
