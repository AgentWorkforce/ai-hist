import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  DatabaseOpenError, NativeContractMismatchError, listSessionCatalogPage, recent, stats,
  validateNativeContract,
} from './index.js';

test('missing database reads are explicit empty cache operations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'relayhistory-missing-db-'));
  const dbPath = join(root, 'missing', 'history.db');
  try {
    assert.deepEqual(await listSessionCatalogPage({ dbPath, limit: 20 }), {
      contractVersion: 1, sessions: [], nextCursor: null,
    });
    assert.deepEqual(await recent({ dbPath, limit: 20 }), []);
    assert.deepEqual(await stats({ dbPath }), {
      total: 0, bySource: {}, byProject: [], firstTimestampMs: null, lastTimestampMs: null,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('SDK/native contract mismatch is actionable', () => {
  assert.throws(
    () => validateNativeContract(999),
    (error: unknown) => error instanceof NativeContractMismatchError
      && error.code === 'NATIVE_CONTRACT_MISMATCH'
      && /requires native contract 2/.test(error.message),
  );
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
