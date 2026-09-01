import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  InvalidArgumentError, SESSION_EVIDENCE_CONTRACT_VERSION,
  getSessionFileEdits, getSessionFileEditsPage, getSessionToolCalls, getSessionToolCallsPage,
  parseStoredJson, sessionFileEdits, sessionToolCalls, sync,
  type EvidenceCursor, type SessionFileEdit, type SessionToolCall,
} from './index.js';

const SHARED_SESSION = 'shared-1';

const CLAUDE_TRANSCRIPT = [
  { type: 'user', uuid: 'u1', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:00.000Z', message: { role: 'user', content: 'update auth' } },
  { type: 'assistant', uuid: 'a1', parentUuid: 'u1', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:01.000Z', message: { role: 'assistant', model: 'claude-test', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Edit', input: { file_path: '/work/app/auth.ts', old_string: 'old', new_string: 'new' } }] } },
  { type: 'user', uuid: 'r1', parentUuid: 'a1', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:02.000Z', message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'ok', toolUseResult: { filePath: '/work/app/auth.ts', structuredPatch: '--- a/auth.ts\n+++ b/auth.ts\n-old\n+new\n', userModified: true } }] } },
  { type: 'assistant', uuid: 'a2', parentUuid: 'r1', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:03.000Z', message: { role: 'assistant', model: 'claude-test', content: [{ type: 'tool_use', id: 'toolu_2', name: 'Write', input: { file_path: '/work/app/notes.md', content: 'notes' } }] } },
  { type: 'user', uuid: 'r2', parentUuid: 'a2', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:04.000Z', message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_2', content: 'ok', toolUseResult: { filePath: '/work/app/notes.md', structuredPatch: [{ oldStart: 1, newStart: 1, lines: ['+notes'] }], userModified: false } }] } },
  { type: 'assistant', uuid: 'a3', parentUuid: 'r2', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:05.000Z', message: { role: 'assistant', model: 'claude-test', content: [{ type: 'tool_use', id: 'toolu_3', name: 'Bash', input: { command: 'cargo test' } }] } },
  { type: 'user', uuid: 'r3', parentUuid: 'a3', sessionId: SHARED_SESSION, cwd: '/work/app', gitBranch: 'main', timestamp: '2026-08-30T10:00:06.000Z', message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_3', is_error: true, content: 'failed' }] } },
];

const CODEX_ROLLOUT = [
  { timestamp: '2026-08-30T11:00:00.000Z', type: 'session_meta', payload: { id: SHARED_SESSION, cwd: '/work/codex', git: { branch: 'main' } } },
  { timestamp: '2026-08-30T11:00:01.000Z', type: 'event_msg', payload: { type: 'user_message', message: 'fix the importer' } },
  { timestamp: '2026-08-30T11:00:02.000Z', type: 'response_item', payload: { type: 'function_call', id: 'fc_1', name: 'exec_command', arguments: '{"cmd":"git status"}', call_id: 'call_1' } },
  { timestamp: '2026-08-30T11:00:03.000Z', type: 'response_item', payload: { type: 'custom_tool_call', id: 'ctc_1', status: 'completed', call_id: 'call_2', name: 'apply_patch', input: '*** Begin Patch\n*** Update File: /work/codex/a.rs\n@@\n+one\n*** End Patch\n' } },
  // One apply_patch call touching two files: file_edits is keyed by tool use,
  // so the writer scopes the key per path rather than losing a file.
  { timestamp: '2026-08-30T11:00:04.000Z', type: 'event_msg', payload: { type: 'patch_apply_end', call_id: 'call_2', success: true, changes: { '/work/codex/a.rs': { type: 'update', unified_diff: '@@\n+one\n-zero' }, '/work/codex/b.rs': { type: 'update', unified_diff: '@@\n+two' } } } },
];

async function seededDatabase(): Promise<{ dbPath: string; cleanup: () => Promise<void> }> {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-evidence-'));
  const home = join(root, 'home');
  const claude = join(home, '.claude', 'projects', 'work-app');
  const codex = join(home, '.codex', 'sessions', '2026', '08', '30');
  await mkdir(claude, { recursive: true });
  await mkdir(codex, { recursive: true });
  await writeFile(join(claude, `${SHARED_SESSION}.jsonl`), `${CLAUDE_TRANSCRIPT.map((line) => JSON.stringify(line)).join('\n')}\n`);
  await writeFile(join(codex, `rollout-${SHARED_SESSION}.jsonl`), `${CODEX_ROLLOUT.map((line) => JSON.stringify(line)).join('\n')}\n`);
  const saved = { HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE };
  process.env.HOME = home;
  process.env.USERPROFILE = home;
  const dbPath = join(root, 'history.db');
  try {
    await sync({ dbPath });
  } finally {
    if (saved.HOME === undefined) delete process.env.HOME; else process.env.HOME = saved.HOME;
    if (saved.USERPROFILE === undefined) delete process.env.USERPROFILE; else process.env.USERPROFILE = saved.USERPROFILE;
  }
  return { dbPath, cleanup: () => rm(root, { recursive: true, force: true }) };
}

test('tool call pages expose structured arguments, errors, and identity', async () => {
  const { dbPath, cleanup } = await seededDatabase();
  try {
    const page = await getSessionToolCallsPage('claude', SHARED_SESSION, { dbPath });
    assert.equal(page.contractVersion, SESSION_EVIDENCE_CONTRACT_VERSION);
    assert.equal(page.source, 'claude');
    assert.equal(page.sessionId, SHARED_SESSION);
    assert.equal(page.nextCursor, null);
    assert.deepEqual(page.toolCalls.map((call) => call.toolUseId), ['toolu_1', 'toolu_2', 'toolu_3']);

    const [edit, write, bash] = page.toolCalls;
    assert.deepEqual(edit.args, { file_path: '/work/app/auth.ts', old_string: 'old', new_string: 'new' });
    assert.deepEqual(JSON.parse(String(edit.argsJson)), edit.args);
    assert.equal(edit.name, 'Edit');
    assert.equal(edit.target, '/work/app/auth.ts');
    assert.equal(edit.messageId, 'a1');
    assert.equal(typeof edit.tsMs, 'number');
    // A tool result that never reported a verdict leaves the error unknown
    // rather than asserting success.
    assert.equal(edit.isError, null);
    assert.equal(write.name, 'Write');
    assert.equal(bash.isError, true);
    assert.deepEqual(bash.args, { command: 'cargo test' });
  } finally {
    await cleanup();
  }
});

test('file edit pages expose patches, provenance, and one row per edited file', async () => {
  const { dbPath, cleanup } = await seededDatabase();
  try {
    const claude = await getSessionFileEditsPage('claude', SHARED_SESSION, { dbPath });
    assert.equal(claude.contractVersion, SESSION_EVIDENCE_CONTRACT_VERSION);
    assert.deepEqual(claude.fileEdits.map((edit) => edit.filePath), ['/work/app/auth.ts', '/work/app/notes.md']);
    const [auth, notes] = claude.fileEdits;
    assert.equal(auth.toolName, 'Edit');
    assert.equal(auth.messageId, 'a1');
    assert.equal(auth.gitBranch, 'main');
    assert.equal(auth.cwd, '/work/app');
    assert.equal(auth.userModified, true);
    assert.equal(auth.linesAdded, 1);
    assert.equal(auth.linesRemoved, 1);
    assert.equal(auth.structuredPatch, '--- a/auth.ts\n+++ b/auth.ts\n-old\n+new\n');
    assert.equal(auth.structuredPatchJson, JSON.stringify('--- a/auth.ts\n+++ b/auth.ts\n-old\n+new\n'));
    assert.equal(notes.userModified, false);
    assert.deepEqual(notes.structuredPatch, [{ oldStart: 1, newStart: 1, lines: ['+notes'] }]);

    // One codex apply_patch call touching two files stores one row per file,
    // keyed by `<call id>#<path>` because file_edits is unique per tool use.
    const codex = await getSessionFileEditsPage('codex', SHARED_SESSION, { dbPath });
    assert.deepEqual(codex.fileEdits.map((edit) => edit.filePath), ['/work/codex/a.rs', '/work/codex/b.rs']);
    assert.deepEqual(codex.fileEdits.map((edit) => edit.toolUseId), ['call_2#/work/codex/a.rs', 'call_2#/work/codex/b.rs']);
    assert.ok(codex.fileEdits.every((edit) => edit.toolName === 'apply_patch'));
  } finally {
    await cleanup();
  }
});

test('evidence pages never mix two providers that share a session id', async () => {
  const { dbPath, cleanup } = await seededDatabase();
  try {
    const claudeCalls = await getSessionToolCalls('claude', SHARED_SESSION, { dbPath });
    const codexCalls = await getSessionToolCalls('codex', SHARED_SESSION, { dbPath });
    assert.ok(claudeCalls.length > 0 && codexCalls.length > 0);
    assert.ok(claudeCalls.every((call) => call.source === 'claude'));
    assert.ok(codexCalls.every((call) => call.source === 'codex'));
    assert.deepEqual(codexCalls.map((call) => call.toolUseId), ['call_1', 'call_2']);

    const claudeEdits = await getSessionFileEdits('claude', SHARED_SESSION, { dbPath });
    const codexEdits = await getSessionFileEdits('codex', SHARED_SESSION, { dbPath });
    assert.ok(claudeEdits.every((edit) => edit.source === 'claude'));
    assert.ok(codexEdits.every((edit) => edit.source === 'codex'));
  } finally {
    await cleanup();
  }
});

test('paged, iterated, and collected reads agree at every page size', async () => {
  const { dbPath, cleanup } = await seededDatabase();
  try {
    const allCalls = await getSessionToolCalls('claude', SHARED_SESSION, { dbPath });
    const allEdits = await getSessionFileEdits('claude', SHARED_SESSION, { dbPath });
    assert.equal(allCalls.length, 3);
    assert.equal(allEdits.length, 2);

    for (const limit of [1, 2, 3, 1000]) {
      const calls: SessionToolCall[] = [];
      for await (const call of sessionToolCalls('claude', SHARED_SESSION, { dbPath, limit })) calls.push(call);
      assert.deepEqual(calls, allCalls, `tool calls at limit ${limit}`);
      const edits: SessionFileEdit[] = [];
      for await (const edit of sessionFileEdits('claude', SHARED_SESSION, { dbPath, limit })) edits.push(edit);
      assert.deepEqual(edits, allEdits, `file edits at limit ${limit}`);
    }

    // Cursors survive the JSON round trip the CLI and MCP server use.
    const first = await getSessionToolCallsPage('claude', SHARED_SESSION, { dbPath, limit: 1 });
    assert.equal(first.toolCalls.length, 1);
    const roundTripped = JSON.parse(JSON.stringify(first.nextCursor)) as EvidenceCursor;
    const second = await getSessionToolCallsPage('claude', SHARED_SESSION, { dbPath, limit: 2, after: roundTripped });
    assert.deepEqual(second.toolCalls, allCalls.slice(1));
    assert.equal(second.nextCursor, null);
  } finally {
    await cleanup();
  }
});

test('evidence pages read unknown sessions empty and require both identity parts', async () => {
  const { dbPath, cleanup } = await seededDatabase();
  try {
    const missing = await getSessionToolCallsPage('claude', 'no-such-session', { dbPath });
    assert.deepEqual(missing.toolCalls, []);
    assert.equal(missing.nextCursor, null);
    assert.deepEqual((await getSessionFileEditsPage('cursor', SHARED_SESSION, { dbPath })).fileEdits, []);

    for (const operation of [
      () => getSessionToolCallsPage('claude', '', { dbPath }),
      () => getSessionFileEditsPage('claude', '   ', { dbPath }),
      () => getSessionToolCallsPage('' as never, SHARED_SESSION, { dbPath }),
    ]) {
      await assert.rejects(operation(), (error: unknown) => error instanceof InvalidArgumentError
        && error.code === 'INVALID_ARGUMENT');
    }
  } finally {
    await cleanup();
  }
});

test('unparseable stored JSON yields null without discarding the raw string', () => {
  assert.deepEqual(parseStoredJson('{"a":1}'), { a: 1 });
  assert.equal(parseStoredJson('not json'), null);
  assert.equal(parseStoredJson(null), null);
  assert.equal(parseStoredJson(undefined), null);
  assert.equal(parseStoredJson('null'), null);
});
