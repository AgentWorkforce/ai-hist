import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  DatabaseOpenError, NativeContractMismatchError, SessionNotFoundError, UnsupportedOperationError,
  discoverSessions, hydrateSession, listSessionCatalogPage, recent, stats, sync,
  validateNativeContract, validateNativeScope,
} from './index.js';

test('missing database reads are explicit empty cache operations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-missing-db-'));
  const dbPath = join(root, 'missing', 'history.db');
  try {
    assert.deepEqual(await listSessionCatalogPage({ dbPath, limit: 20 }), {
      contractVersion: 2, scope: 'local', sessions: [], nextCursor: null,
    });
    assert.deepEqual(await recent({ dbPath, limit: 20 }), []);
    assert.deepEqual(await stats({ dbPath }), {
      scope: 'local', total: 0, bySource: {}, byProject: [], firstTimestampMs: null, lastTimestampMs: null,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('targeted hydration requires an existing catalog row', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-hydrate-missing-'));
  try {
    await assert.rejects(
      hydrateSession({ source: 'claude', sessionId: 'missing', dbPath: join(root, 'history.db') }),
      (error: unknown) => error instanceof SessionNotFoundError
        && error.code === 'SESSION_NOT_FOUND'
        && /discoverSessions/.test(error.message),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('SDK/native contract mismatch is actionable', () => {
  assert.throws(
    () => validateNativeContract(999),
    (error: unknown) => error instanceof NativeContractMismatchError
      && error.code === 'NATIVE_CONTRACT_MISMATCH'
      && /requires native contract 4/.test(error.message),
  );
  assert.throws(
    () => validateNativeScope('cloud'),
    (error: unknown) => error instanceof NativeContractMismatchError
      && error.code === 'NATIVE_CONTRACT_MISMATCH'
      && /invalid session scope/.test(error.message),
  );
});

test('unconfigured remote acquisition has one stable SDK error', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-unsupported-remote-'));
  const dbPath = join(root, 'history.db');
  // Connector detection reads the provider CLIs' stored sign-ins under HOME,
  // so point it at an empty home rather than the machine running the tests.
  const saved = { HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE };
  process.env.HOME = root;
  process.env.USERPROFILE = root;
  try {
    for (const operation of [
      () => discoverSessions({ dbPath, scope: 'remote' }),
      () => sync({ dbPath, scope: 'remote' }),
    ]) {
      await assert.rejects(
        operation(),
        (error: unknown) => error instanceof UnsupportedOperationError
          && error.code === 'UNSUPPORTED_OPERATION'
          && error.message.includes('no remote provider connectors are configured'),
      );
    }
  } finally {
    if (saved.HOME === undefined) delete process.env.HOME; else process.env.HOME = saved.HOME;
    if (saved.USERPROFILE === undefined) delete process.env.USERPROFILE; else process.env.USERPROFILE = saved.USERPROFILE;
    await rm(root, { recursive: true, force: true });
  }
});

test('database open failures use the stable SDK error', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'relayhistory-not-a-db-'));
  try {
    for (const operation of [
      () => recent({ dbPath: directory }),
      () => listSessionCatalogPage({ dbPath: directory }),
    ]) {
      await assert.rejects(
        operation(),
        (error: unknown) => error instanceof DatabaseOpenError
          && error.code === 'DATABASE_OPEN_FAILED'
          && error.message.includes(directory),
      );
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
