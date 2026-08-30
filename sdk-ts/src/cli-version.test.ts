import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
const metadata = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8')) as { version: string };
const cli = new URL('./cli.js', import.meta.url);

for (const args of [['--version'], ['-V'], ['--version', '--no-warning']]) {
  test(`CLI ${args.join(' ')} reports the npm package version`, async () => {
    const result = await execFileAsync(process.execPath, [cli.pathname, ...args], {
      env: { ...process.env, RELAYHISTORY_NO_UPDATE_CHECK: '1' },
    });
    assert.equal(result.stdout.trim(), `ai-hist ${metadata.version}`);
    assert.equal(result.stderr, '');
  });
}
