import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const sourceDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'src');
const repositoryRoot = join(sourceDir, '..', '..');

test('production TypeScript has one native implementation', async () => {
  const files = ['index.ts', 'cli.ts', 'mcp-server.ts'];
  const source = (await Promise.all(files.map((file) => readFile(join(sourceDir, file), 'utf8')))).join('\n');
  for (const forbidden of ['sql.js', 'node:child_process', 'AI_HIST_RUST_BIN', "fallback: 'jsonl'", 'readFile(dbPath)']) {
    assert.equal(source.includes(forbidden), false, `production source contains ${forbidden}`);
  }
  const pkg = JSON.parse(await readFile(join(repositoryRoot, 'sdk-ts', 'package.json'), 'utf8')) as { dependencies: Record<string, string> };
  assert.equal(pkg.dependencies['sql.js'], undefined);
  assert.equal(typeof pkg.dependencies['ai-hist-native'], 'string');
});

test('CLI and MCP import only the public SDK for history operations', async () => {
  const [cli, mcp] = await Promise.all([
    readFile(join(sourceDir, 'cli.ts'), 'utf8'),
    readFile(join(sourceDir, 'mcp-server.ts'), 'utf8'),
  ]);
  assert.match(cli, /from '\.\/index\.js'/);
  assert.match(mcp, /from '\.\/index\.js'/);
  assert.doesNotMatch(cli + mcp, /ai-hist-native|node:sqlite|sql\.js|child_process/);
});
