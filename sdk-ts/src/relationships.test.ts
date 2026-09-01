import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  InvalidArgumentError,
  discoverSessions, getSessionChildrenPage, getSessionRelationships, getSessionTree,
  hydrateSession, sessionDescendants, sessionEventsIncludingDescendants, sync,
  type SessionRelationship, type SessionTreeNode,
} from './index.js';

/** Runs one case against a private HOME so provider scans see only its fixtures. */
async function withHome(
  prefix: string,
  body: (home: string, dbPath: string) => Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), prefix));
  const home = join(root, 'home');
  await mkdir(home, { recursive: true });
  const saved = { HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE };
  process.env.HOME = home;
  process.env.USERPROFILE = home;
  try {
    await body(home, join(root, 'history.db'));
  } finally {
    if (saved.HOME === undefined) delete process.env.HOME; else process.env.HOME = saved.HOME;
    if (saved.USERPROFILE === undefined) delete process.env.USERPROFILE; else process.env.USERPROFILE = saved.USERPROFILE;
    await rm(root, { recursive: true, force: true });
  }
}

async function codexRollout(
  home: string,
  id: string,
  options: { at: string; parent?: string; label?: string },
): Promise<void> {
  const day = join(home, '.codex', 'sessions', '2026', '08', '31');
  await mkdir(day, { recursive: true });
  const payload = options.parent === undefined
    ? { id, cwd: '/work/app' }
    : {
      id, cwd: '/work/app', session_id: options.parent, parent_thread_id: options.parent,
      thread_source: 'subagent', source: { subagent: { other: options.label ?? 'guardian' } },
    };
  await writeFile(join(day, `rollout-${id}.jsonl`), [
    JSON.stringify({ timestamp: options.at, type: 'session_meta', payload }),
    JSON.stringify({ timestamp: options.at, type: 'event_msg', payload: { type: 'user_message', message: `${id} prompt` } }),
    JSON.stringify({ timestamp: options.at, type: 'event_msg', payload: { type: 'agent_message', message: `${id} answer` } }),
    '',
  ].join('\n'));
}

function childIds(children: SessionRelationship[]): Array<string | null> {
  return children.map((child) => child.childSessionId);
}

test('missing databases answer relationship queries with provider capabilities', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-relationships-empty-'));
  const dbPath = join(root, 'missing', 'history.db');
  try {
    assert.deepEqual(await getSessionRelationships({ source: 'codex', sessionId: 'root', dbPath }), {
      contractVersion: 1, source: 'codex', sessionId: 'root', asParent: [], asChild: [],
      capabilities: {
        source: 'codex', stableChildIdentity: 'always', recordsAgentType: true,
        recordsSpawnTime: true, recordsEvidenceLocator: true,
      },
      diagnostics: [],
    });
    assert.deepEqual(await getSessionTree({ source: 'claude', sessionId: 'root', dbPath }), {
      contractVersion: 1, source: 'claude', rootSessionId: 'root', nodes: [], unlinked: [],
      capabilities: {
        source: 'claude', stableChildIdentity: 'sometimes', recordsAgentType: true,
        recordsSpawnTime: true, recordsEvidenceLocator: true,
      },
      diagnostics: [], truncated: false, maxDepthReached: 0,
    });
    assert.deepEqual(await getSessionChildrenPage({ source: 'cursor', sessionId: 'root', dbPath }), {
      children: [], nextCursor: null,
    });
    const cursor = await getSessionRelationships({ source: 'cursor', sessionId: 'root', dbPath });
    assert.equal(cursor.capabilities.stableChildIdentity, 'never');
    assert.equal(cursor.capabilities.recordsEvidenceLocator, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('relationship queries reject invalid identities before reaching the engine', async () => {
  const dbPath = join(tmpdir(), 'relayhistory-never-created.db');
  for (const operation of [
    () => getSessionRelationships({ source: 'trajectory' as never, sessionId: 'root', dbPath }),
    () => getSessionTree({ source: 'nope' as never, sessionId: 'root', dbPath }),
    () => getSessionChildrenPage({ source: 'codex', sessionId: '  ', dbPath }),
    () => getSessionTree({ source: 'codex', sessionId: '', dbPath }),
    async () => { for await (const _ of sessionDescendants({ source: 'codex', sessionId: '', dbPath })) break; },
    async () => {
      for await (const _ of sessionEventsIncludingDescendants({ source: 'bad' as never, sessionId: 'root', dbPath })) break;
    },
  ]) {
    await assert.rejects(operation(), (error: unknown) => error instanceof InvalidArgumentError
      && error.code === 'INVALID_ARGUMENT');
  }
});

test('codex delegation topology round-trips through discovery and hydration', async () => {
  await withHome('relayhistory-relationships-codex-', async (home, dbPath) => {
    await codexRollout(home, 'root', { at: '2026-08-31T10:00:00Z' });
    await codexRollout(home, 'child-a', { at: '2026-08-31T10:00:03Z', parent: 'root', label: 'guardian' });
    await codexRollout(home, 'child-b', { at: '2026-08-31T10:00:02Z', parent: 'root', label: 'reviewer' });

    await discoverSessions({ dbPath, sources: ['codex'] });
    const hydrated = await hydrateSession({ source: 'codex', sessionId: 'root', dbPath });
    assert.deepEqual([...hydrated.relatedSessionIds].sort(), ['child-a', 'child-b']);

    const relationships = await getSessionRelationships({ source: 'codex', sessionId: 'root', dbPath });
    assert.equal(relationships.contractVersion, 1);
    assert.deepEqual(relationships.asChild, []);
    // Spawn time orders children ahead of the relationship uid tiebreaker.
    assert.deepEqual(childIds(relationships.asParent), ['child-b', 'child-a']);
    const [reviewer, guardian] = relationships.asParent;
    assert.equal(reviewer.identityStatus, 'observed');
    assert.equal(reviewer.relationship, 'delegated');
    assert.equal(reviewer.evidenceKind, 'codex_session_meta');
    assert.equal(reviewer.evidenceRef, 'root');
    assert.equal(reviewer.childAgentType, 'reviewer');
    assert.equal(reviewer.childHasEvents, true);
    assert.equal(reviewer.relationshipUid, 'child:child-b');
    assert.ok(reviewer.evidenceLocator?.endsWith('rollout-child-b.jsonl'));
    assert.equal(typeof reviewer.spawnedAtMs, 'number');
    assert.ok((guardian.spawnedAtMs ?? 0) > (reviewer.spawnedAtMs ?? 0));

    // The same edge is visible from the child, in the other direction.
    const child = await getSessionRelationships({ source: 'codex', sessionId: 'child-a', dbPath });
    assert.deepEqual(child.asParent, []);
    assert.deepEqual(child.asChild, [guardian]);

    const tree = await getSessionTree({ source: 'codex', sessionId: 'root', dbPath });
    assert.deepEqual(tree.nodes.map((node) => [node.sessionId, node.depth]), [
      ['root', 0], ['child-b', 1], ['child-a', 1],
    ]);
    assert.equal(tree.nodes[0].childCount, 2);
    assert.equal(tree.nodes[0].relationship, null);
    assert.equal(tree.nodes[1].parentSessionId, 'root');
    assert.equal(tree.nodes[1].hasEvents, true);
    assert.equal(tree.maxDepthReached, 1);
    assert.equal(tree.truncated, false);
    assert.deepEqual(tree.unlinked, []);
    assert.deepEqual(tree.diagnostics, []);

    const first = await getSessionChildrenPage({ source: 'codex', sessionId: 'root', dbPath, limit: 1 });
    assert.deepEqual(childIds(first.children), ['child-b']);
    assert.equal(first.nextCursor?.relationshipUid, 'child:child-b');
    const second = await getSessionChildrenPage({
      source: 'codex', sessionId: 'root', dbPath, limit: 1, after: first.nextCursor ?? undefined,
    });
    assert.deepEqual(childIds(second.children), ['child-a']);
  });
});

test('tree traversal is deterministic, depth-bounded, and cycle-safe', async () => {
  await withHome('relayhistory-relationships-deep-', async (home, dbPath) => {
    await codexRollout(home, 'root', { at: '2026-08-31T10:00:00Z' });
    await codexRollout(home, 'child', { at: '2026-08-31T10:00:01Z', parent: 'root' });
    await codexRollout(home, 'grandchild', { at: '2026-08-31T10:00:02Z', parent: 'child' });
    // Two threads that name each other are recorded honestly and must not
    // make traversal loop.
    await codexRollout(home, 'loop-a', { at: '2026-08-31T10:00:03Z', parent: 'loop-b' });
    await codexRollout(home, 'loop-b', { at: '2026-08-31T10:00:04Z', parent: 'loop-a' });
    await sync({ dbPath });

    const tree = await getSessionTree({ source: 'codex', sessionId: 'root', dbPath });
    assert.deepEqual(tree.nodes.map((node) => node.sessionId), ['root', 'child', 'grandchild']);
    assert.equal(tree.maxDepthReached, 2);
    assert.deepEqual(
      await getSessionTree({ source: 'codex', sessionId: 'root', dbPath }),
      tree,
      'repeated traversals of the same database are identical',
    );

    const shallow = await getSessionTree({ source: 'codex', sessionId: 'root', dbPath, maxDepth: 1 });
    assert.deepEqual(shallow.nodes.map((node) => node.sessionId), ['root', 'child']);
    assert.equal(shallow.nodes[1].truncated, true);
    assert.equal(shallow.truncated, true);
    assert.deepEqual(shallow.diagnostics.map((item) => item.code), ['RELATIONSHIP_TREE_DEPTH_LIMIT']);

    const cycle = await getSessionTree({ source: 'codex', sessionId: 'loop-a', dbPath });
    assert.deepEqual(cycle.nodes.map((node) => node.sessionId), ['loop-a', 'loop-b']);
    assert.equal(cycle.truncated, true);
    assert.deepEqual(cycle.diagnostics.map((item) => item.code), ['RELATIONSHIP_CYCLE']);

    // A different source never sees these edges.
    assert.deepEqual((await getSessionTree({ source: 'claude', sessionId: 'root', dbPath })).nodes.map((node) => node.sessionId), ['root']);
  });
});

test('sessionDescendants walks the same descendants as the materialized tree', async () => {
  await withHome('relayhistory-relationships-walk-', async (home, dbPath) => {
    await codexRollout(home, 'root', { at: '2026-08-31T10:00:00Z' });
    await codexRollout(home, 'child-a', { at: '2026-08-31T10:00:01Z', parent: 'root' });
    await codexRollout(home, 'child-b', { at: '2026-08-31T10:00:02Z', parent: 'root' });
    await codexRollout(home, 'grandchild', { at: '2026-08-31T10:00:03Z', parent: 'child-a' });
    await sync({ dbPath });

    const walked: SessionTreeNode[] = [];
    // A page limit of one forces the walker through its cursor path.
    for await (const node of sessionDescendants({ source: 'codex', sessionId: 'root', dbPath, pageLimit: 1 })) {
      walked.push(node);
    }
    const tree = await getSessionTree({ source: 'codex', sessionId: 'root', dbPath });
    const byId = (nodes: SessionTreeNode[]) => [...nodes].sort((left, right) => left.sessionId.localeCompare(right.sessionId));
    assert.deepEqual(byId(walked), byId(tree.nodes.slice(1)));

    const bounded: string[] = [];
    for await (const node of sessionDescendants({ source: 'codex', sessionId: 'root', dbPath, maxDepth: 1 })) {
      bounded.push(node.sessionId);
    }
    assert.deepEqual(bounded.sort(), ['child-a', 'child-b']);
  });
});

test('descendant event iteration preserves each event\'s owning session', async () => {
  await withHome('relayhistory-relationships-events-', async (home, dbPath) => {
    await codexRollout(home, 'root', { at: '2026-08-31T10:00:00Z' });
    await codexRollout(home, 'child', { at: '2026-08-31T10:00:01Z', parent: 'root' });
    await codexRollout(home, 'grandchild', { at: '2026-08-31T10:00:02Z', parent: 'child' });
    await sync({ dbPath });

    const owners: string[] = [];
    for await (const event of sessionEventsIncludingDescendants({ source: 'codex', sessionId: 'root', dbPath })) {
      assert.equal(event.source, 'codex');
      owners.push(event.sessionId);
    }
    assert.deepEqual([...new Set(owners)], ['root', 'child', 'grandchild']);
    assert.ok(owners.filter((owner) => owner === 'child').length >= 2);

    const descendantsOnly: string[] = [];
    for await (const event of sessionEventsIncludingDescendants({
      source: 'codex', sessionId: 'root', dbPath, includeRoot: false,
    })) {
      descendantsOnly.push(event.sessionId);
    }
    assert.equal(descendantsOnly.includes('root'), false);
    assert.deepEqual([...new Set(descendantsOnly)], ['child', 'grandchild']);
  });
});

test('claude evidence without a stable child identity stays unlinked', async () => {
  await withHome('relayhistory-relationships-claude-', async (home, dbPath) => {
    const project = join(home, '.claude', 'projects', 'app');
    await mkdir(project, { recursive: true });
    await writeFile(join(project, 'session-1.jsonl'), [
      JSON.stringify({
        sessionId: 'session-1', uuid: 'u1', cwd: '/work/app', type: 'user',
        message: { role: 'user', content: 'first prompt' }, timestamp: '2026-08-31T10:00:00Z',
      }),
      '',
    ].join('\n'));
    // This provider version records the sidechain but never names the child.
    await writeFile(join(project, 'agent-child.jsonl'), [
      JSON.stringify({
        sessionId: 'session-1', uuid: 'side-u', isSidechain: true, type: 'user',
        message: { role: 'user', content: 'delegated instruction' }, timestamp: '2026-08-31T10:00:01Z',
      }),
      JSON.stringify({
        sessionId: 'session-1', uuid: 'side-a', isSidechain: true, type: 'assistant',
        message: { role: 'assistant', content: 'side result' }, timestamp: '2026-08-31T10:00:02Z',
      }),
      '',
    ].join('\n'));

    await discoverSessions({ dbPath, sources: ['claude'] });
    const hydrated = await hydrateSession({ source: 'claude', sessionId: 'session-1', dbPath });
    assert.deepEqual(hydrated.relatedSessionIds, []);
    assert.ok(hydrated.diagnostics.some((item) => item.code === 'RELATIONSHIP_UNLINKED_CHILD'));

    const relationships = await getSessionRelationships({ source: 'claude', sessionId: 'session-1', dbPath });
    assert.equal(relationships.capabilities.stableChildIdentity, 'sometimes');
    assert.equal(relationships.asParent.length, 1);
    const [evidence] = relationships.asParent;
    assert.equal(evidence.childSessionId, null);
    assert.equal(evidence.identityStatus, 'unlinked');
    assert.equal(evidence.evidenceKind, 'claude_sidechain_records');
    assert.equal(evidence.childHasEvents, false);
    // The child id is never taken from the file name.
    assert.equal(evidence.relationshipUid.startsWith('evidence:'), true);
    assert.ok(evidence.evidenceLocator?.endsWith('agent-child.jsonl'));

    const tree = await getSessionTree({ source: 'claude', sessionId: 'session-1', dbPath });
    assert.deepEqual(tree.nodes.map((node) => node.sessionId), ['session-1']);
    assert.deepEqual(tree.unlinked, relationships.asParent);
    assert.deepEqual(tree.diagnostics.map((item) => item.code), ['RELATIONSHIP_UNLINKED_CHILD']);
  });
});

test('large trees stay bounded by the node budget', async () => {
  await withHome('relayhistory-relationships-bounded-', async (home, dbPath) => {
    await codexRollout(home, 'root', { at: '2026-08-31T10:00:00Z' });
    for (let index = 0; index < 200; index++) {
      await codexRollout(home, `child-${String(index).padStart(3, '0')}`, {
        at: '2026-08-31T10:00:01Z', parent: 'root',
      });
    }
    await sync({ dbPath });

    const tree = await getSessionTree({ source: 'codex', sessionId: 'root', dbPath, maxNodes: 50 });
    assert.equal(tree.nodes.length, 50);
    assert.equal(tree.truncated, true);
    assert.deepEqual(tree.diagnostics.map((item) => item.code), ['RELATIONSHIP_TREE_TRUNCATED']);
    assert.equal(tree.nodes[0].childCount, 200);

    const page = await getSessionChildrenPage({ source: 'codex', sessionId: 'root', dbPath, limit: 100 });
    assert.equal(page.children.length, 100);
    assert.equal(page.nextCursor?.relationshipUid, 'child:child-099');
  });
});
