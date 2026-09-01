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

test('sessions hydrate uses the SDK contract and is idempotent', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-hydrate-'));
  const home = join(root, 'home');
  const claude = join(home, '.claude', 'projects', 'project');
  const db = join(root, 'history.db');
  await mkdir(claude, { recursive: true });
  await writeFile(join(claude, 'claude-hydrate.jsonl'), `${JSON.stringify({
    sessionId: 'claude-hydrate', uuid: 'user-1', cwd: '/work/claude', type: 'user',
    message: { role: 'user', content: 'hydrate this session' }, timestamp: '2026-08-30T10:00:00.000Z',
  })}\n`);
  const env = { ...process.env, HOME: home, USERPROFILE: home };
  try {
    await run(process.execPath, [
      cli, 'sessions', 'discover', '--source', 'claude', '--db', db, '--no-warning',
    ], { env });
    const first = await run(process.execPath, [
      cli, 'sessions', 'hydrate', 'claude', 'claude-hydrate', '--db', db, '--json', '--no-warning',
    ], { env });
    const hydrated = JSON.parse(first.stdout) as Record<string, unknown>;
    assert.equal(hydrated.contract_version, 1);
    assert.equal(hydrated.status, 'hydrated');
    assert.equal(hydrated.discovery_state, 'full');
    assert.deepEqual(hydrated.evidence, {
      prompts: 1, events: 1, tool_calls: 0, related_sessions: 0,
    });

    const second = await run(process.execPath, [
      cli, 'sessions', 'hydrate', 'claude', 'claude-hydrate', '--db', db, '--json', '--no-warning',
    ], { env });
    assert.equal((JSON.parse(second.stdout) as Record<string, unknown>).status, 'unchanged');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('sessions relationships and sessions tree render topology in both modes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-cli-topology-'));
  const home = join(root, 'home');
  const day = join(home, '.codex', 'sessions', '2026', '08', '31');
  const db = join(root, 'history.db');
  await mkdir(day, { recursive: true });
  const rollout = (id: string, at: string, parent?: string) => [
    JSON.stringify({
      timestamp: at, type: 'session_meta',
      payload: parent === undefined
        ? { id, cwd: '/work/app' }
        : {
          id, cwd: '/work/app', session_id: parent, parent_thread_id: parent,
          thread_source: 'subagent', source: { subagent: { other: 'guardian' } },
        },
    }),
    JSON.stringify({ timestamp: at, type: 'event_msg', payload: { type: 'user_message', message: `${id} prompt` } }),
    JSON.stringify({ timestamp: at, type: 'event_msg', payload: { type: 'agent_message', message: `${id} answer` } }),
    '',
  ].join('\n');
  await writeFile(join(day, 'rollout-topology-root.jsonl'), rollout('topology-root', '2026-08-31T10:00:00Z'));
  await writeFile(join(day, 'rollout-topology-child.jsonl'), rollout('topology-child', '2026-08-31T10:00:01Z', 'topology-root'));
  const env = { ...process.env, HOME: home, USERPROFILE: home };
  try {
    await run(process.execPath, [cli, 'sessions', 'discover', '--source', 'codex', '--db', db, '--no-warning'], { env });
    await run(process.execPath, [cli, 'sessions', 'hydrate', 'codex', 'topology-root', '--db', db, '--no-warning'], { env });

    const relationships = await run(process.execPath, [
      cli, 'sessions', 'relationships', 'codex', 'topology-root', '--db', db, '--json', '--no-warning',
    ], { env });
    const wire = JSON.parse(relationships.stdout) as Record<string, unknown>;
    assert.equal(wire.contract_version, 1);
    assert.equal(wire.session_id, 'topology-root');
    assert.deepEqual(wire.as_child, []);
    const [edge] = wire.as_parent as Array<Record<string, unknown>>;
    assert.equal(edge.child_session_id, 'topology-child');
    assert.equal(edge.identity_status, 'observed');
    assert.equal(edge.child_has_events, true);
    assert.equal(edge.relationship_uid, 'child:topology-child');
    assert.equal((wire.capabilities as Record<string, unknown>).stable_child_identity, 'always');

    const readable = await run(process.execPath, [
      cli, 'sessions', 'relationships', 'codex', 'topology-root', '--db', db, '--no-warning',
    ], { env });
    assert.match(readable.stdout, /codex\/topology-root: 1 child relationship\(s\), 0 parent relationship\(s\)/);
    assert.match(readable.stdout, /child {2}topology-child {2}delegated {2}guardian/);
    assert.match(readable.stdout, /identity=observed/);
    assert.match(readable.stdout, /capability: stable child identity = always/);

    const empty = await run(process.execPath, [
      cli, 'sessions', 'relationships', 'codex', 'topology-child', '--db', db, '--no-warning',
    ], { env });
    assert.match(empty.stdout, /codex\/topology-child: 0 child relationship\(s\), 1 parent relationship\(s\)/);
    assert.match(empty.stdout, /parent {2}topology-root/);

    const treeJson = await run(process.execPath, [
      cli, 'sessions', 'tree', 'codex', 'topology-root', '--db', db, '--json', '--no-warning',
    ], { env });
    const tree = JSON.parse(treeJson.stdout) as Record<string, unknown>;
    assert.equal(tree.root_session_id, 'topology-root');
    assert.equal(tree.truncated, false);
    assert.deepEqual((tree.nodes as Array<Record<string, unknown>>).map((node) => node.session_id), [
      'topology-root', 'topology-child',
    ]);

    const treeHuman = await run(process.execPath, [
      cli, 'sessions', 'tree', 'codex', 'topology-root', '--max-depth', '4', '--db', db, '--no-warning',
    ], { env });
    assert.match(treeHuman.stdout, /^topology-root\n {2}topology-child {2}\[delegated guardian] {2}events=yes\n1 descendant\(s\), max depth 1\n$/);

    await assert.rejects(
      run(process.execPath, [
        cli, 'sessions', 'tree', 'codex', 'topology-root', '--max-depth', '0', '--db', db, '--no-warning',
      ], { env }),
      (error: unknown) => typeof error === 'object' && error !== null
        && 'code' in error && error.code === 1
        && 'stderr' in error && String(error.stderr).includes('INVALID_ARGUMENT'),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('topology commands require SOURCE and SESSION_ID and reject location scope', async () => {
  for (const subcommand of ['relationships', 'tree']) {
    await assert.rejects(
      run(process.execPath, [cli, 'sessions', subcommand, '--no-warning']),
      (error: unknown) => isUsageFailure(error, `sessions ${subcommand} requires SOURCE and SESSION_ID`),
    );
    await assert.rejects(
      run(process.execPath, [cli, 'sessions', subcommand, 'codex', 'root', 'extra', '--no-warning']),
      (error: unknown) => isUsageFailure(error, 'does not accept positional argument'),
    );
    await assert.rejects(
      run(process.execPath, [cli, 'sessions', subcommand, 'codex', 'root', '--remote', '--no-warning']),
      (error: unknown) => isUsageFailure(error, 'does not accept --local, --remote, or --all'),
    );
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
