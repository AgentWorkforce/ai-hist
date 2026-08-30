import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  DatabaseOpenError, NativeContractMismatchError, recent, validateNativeContract,
} from './index.js';

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
    await assert.rejects(
      recent({ dbPath: directory }),
      (error: unknown) => error instanceof DatabaseOpenError
        && error.code === 'DATABASE_OPEN_FAILED'
        && error.message.includes(directory),
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
