/**
 * Session catalog: the materialized shallow index of coding-agent sessions.
 *
 * Two halves, deliberately split by who owns the write:
 *
 *   - **Cache-only listing** — `AiHist.listSessionCatalog()` in `index.ts`
 *     reads the `sessions` table and nothing else. No provider file is opened,
 *     no `history` / `session_events` scan happens. This module supplies the
 *     row type and the raw-row → `CatalogSession` mapping it uses.
 *   - **Shallow discovery** — `discoverSessions()` here drives
 *     `ai-hist sessions discover --json`, which scans the known provider
 *     locations with bounded reads and upserts the catalog. It is a top-level
 *     function rather than an `AiHist` method because discovery *writes* the
 *     on-disk database, and an `AiHist` instance is an in-memory snapshot taken
 *     at open time — a method would hand back a reader that cannot see the rows
 *     it just wrote. Re-open (or `openAiHist` again) afterwards to read them.
 *
 * The Rust CLI stays the single source of truth for discovery (provider
 * locations, bounded reads, global recency ordering, stamp comparison), exactly
 * as `cloud-push.ts` defers to `ai-hist push` for the push pipeline.
 */

import { spawn, type SpawnOptions } from 'node:child_process';
import { StringDecoder } from 'node:string_decoder';

import { resolveAiHistBinary } from './cloud-push.js';
import type { Source } from './index.js';

/**
 * Version of the session-catalog output contract this SDK understands.
 * Mirrors the native `SESSION_CATALOG_CONTRACT_VERSION`; a `summary` whose
 * `contractVersion` differs came from a binary that speaks a different dialect.
 */
export const SESSION_CATALOG_CONTRACT_VERSION = 1;

/**
 * One row of the session catalog — a coding-agent session as shallow
 * discovery knows it.
 *
 * Most fields are *observed* (read straight out of provider data);
 * `firstPrompt` is *derived* — a bounded excerpt of the first substantive human
 * prompt. Provider quirks worth knowing:
 *
 *   - `cursor` rows never have a `firstActivityMs` (the provider records no
 *     timestamps) and their `lastActivityMs` comes from the file mtime.
 *   - `relay` rows never have a `cwd` (a relay thread has no working
 *     directory) and only exist for threads a previous sync pulled down.
 *   - `lastAssistantText` is only written by a full ingest (`ai-hist sync`), so
 *     it is `null` on rows that have only ever been discovered shallowly.
 *   - `models` / `workspaceRoots` are best effort: `[]` means "not seen in a
 *     bounded read", not "none exist".
 */
export interface CatalogSession {
  /** Provider that owns the session. `trajectory` never appears here. */
  source: Source;
  /** The provider's own session id. `(source, sessionId)` is the catalog key. */
  sessionId: string;
  cwd: string | null;
  gitBranch: string | null;
  firstActivityMs: number | null;
  lastActivityMs: number | null;
  /** Derived: bounded excerpt of the first substantive human prompt. */
  firstPrompt: string | null;
  /** Full-ingest only; `null` on shallow-only rows. */
  lastAssistantText: string | null;
  /** Parsed from `models_json`; `[]` when nothing was observed. */
  models: string[];
  originator: string | null;
  agentVersion: string | null;
  repoUrl: string | null;
  initialCommit: string | null;
  /** Parsed from `workspace_roots_json`; `[]` when nothing was observed. */
  workspaceRoots: string[];
  rawPath: string | null;
  /** Change stamp of the raw source at scan time (`v{scanner}:{provider stamp}`). */
  sourceStamp: string | null;
  /**
   * `'shallow'` (catalog row only) or `'full'` (full evidence ingested).
   * Rows written before the catalog existed store `NULL` and are reported as
   * `'full'`, matching the native reader.
   */
  discoveryState: 'shallow' | 'full';
  /** `true` when the row was served from the catalog without re-reading the source. */
  fromCache: boolean;
  /** Parser generation that wrote the row; absent on rows streamed by discovery. */
  parserVersion?: number;
}

/** Filters for the cache-only catalog listing. */
export interface ListCatalogOptions {
  /** Restrict to these sources. Omit for every discoverable source. */
  sources?: Iterable<Source | string>;
  /** Row cap. Default 50, mirroring the native `DEFAULT_CATALOG_LIMIT`. */
  limit?: number;
  /** Keyset pagination: only sessions strictly older than this epoch-ms cutoff. */
  beforeMs?: number;
}

/** A non-fatal failure during discovery: one provider, or one bad session. */
export interface DiscoveryDiagnostic {
  source: string;
  /** File path or provider-scoped handle, when the failure has one. */
  locator: string | null;
  error: string;
}

/** Per-provider tallies for one discovery run. */
export interface ProviderDiscoverySummary {
  /** Candidates enumerated, before the global limit. */
  candidates: number;
  /** Rows emitted after a shallow read. */
  discovered: number;
  /** Rows served from the catalog because the stamp was unchanged. */
  skippedUnchanged: number;
  /** `true` when enumeration itself failed. */
  failed: boolean;
}

/** Work one discovery run actually performed — bounded-work evidence. */
export interface DiscoveryCounters {
  candidatesEnumerated: number;
  shallowReads: number;
  skippedUnchanged: number;
  filesOpened: number;
  bytesRead: number;
}

/** A source that deliberately has no shallow adapter, and why. */
export interface SourceExemption {
  source: string;
  reason: string;
}

/** The closing `summary` line of a discovery run. */
export interface DiscoverySummary {
  /** Compare against {@link SESSION_CATALOG_CONTRACT_VERSION}. */
  contractVersion: number;
  discovered: number;
  skippedUnchanged: number;
  /** Per-provider tallies, keyed by source name. */
  providers: Record<string, ProviderDiscoverySummary>;
  exemptSources: SourceExemption[];
  counters: DiscoveryCounters;
}

/** Everything one `discoverSessions()` run produced. */
export interface DiscoverResult {
  /** Rows in the order the CLI streamed them: newest first, globally. */
  sessions: CatalogSession[];
  /** Non-fatal per-provider / per-session failures. Never throws on these. */
  diagnostics: DiscoveryDiagnostic[];
  /** `null` only when the binary produced no summary line (unexpected). */
  summary: DiscoverySummary | null;
}

export interface DiscoverSessionsOptions {
  /** Explicit path to the ai-hist binary (overrides discovery). */
  binPath?: string;
  /** Restrict to these sources (forwarded as repeated `--source`). */
  sources?: Iterable<Source | string>;
  /**
   * Global cap across all providers, applied by recency (`--limit`). The
   * native ordering is global, so a limit of 3 returns the three newest
   * sessions overall, not three per provider.
   */
  limit?: number;
  /** Called for each session row as it streams, before the promise resolves. */
  onSession?: (session: CatalogSession) => void;
  /** Called for each diagnostic line as it streams. */
  onDiagnostic?: (diagnostic: DiscoveryDiagnostic) => void;
  /** Extra environment for the spawned binary (merged over `process.env`). */
  env?: NodeJS.ProcessEnv;
  /** Injectable spawn for testing. */
  spawnFn?: typeof spawn;
}

const AI_HIST_RUST_BIN_ENV = 'AI_HIST_RUST_BIN';
/** Spawn failures that all mean the same thing: no usable binary. */
const MISSING_BINARY_CODES = new Set(['ENOENT', 'EACCES', 'EPERM', 'ENOTDIR', 'EISDIR']);

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function count(value: unknown): number {
  return num(value) ?? 0;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

/**
 * Parse a JSON string array column (`models_json`, `workspace_roots_json`).
 * `NULL`, `''`, and malformed JSON all mean "nothing observed" → `[]`; the
 * column never stores `[]` itself.
 */
export function parseJsonStringList(raw: string | null | undefined): string[] {
  if (typeof raw !== 'string' || raw.length === 0) return [];
  try {
    return stringList(JSON.parse(raw));
  } catch {
    return [];
  }
}

/** `NULL` `discovery_state` predates the catalog and means a fully ingested row. */
export function normalizeDiscoveryState(raw: unknown): 'shallow' | 'full' {
  return raw === 'shallow' ? 'shallow' : 'full';
}

/** Map one snake_case CLI/`sessions`-row object onto a {@link CatalogSession}. */
export function catalogSessionFromJson(raw: Record<string, unknown>): CatalogSession {
  return {
    source: (str(raw.source) ?? '') as Source,
    sessionId: str(raw.session_id) ?? '',
    cwd: str(raw.cwd),
    gitBranch: str(raw.git_branch),
    firstActivityMs: num(raw.first_activity_ms),
    lastActivityMs: num(raw.last_activity_ms),
    firstPrompt: str(raw.first_prompt),
    lastAssistantText: str(raw.last_assistant_text),
    // Discovery emits parsed arrays; a raw table row carries the JSON text.
    models: Array.isArray(raw.models) ? stringList(raw.models) : parseJsonStringList(str(raw.models_json)),
    originator: str(raw.originator),
    agentVersion: str(raw.agent_version),
    repoUrl: str(raw.repo_url),
    initialCommit: str(raw.initial_commit),
    workspaceRoots: Array.isArray(raw.workspace_roots)
      ? stringList(raw.workspace_roots)
      : parseJsonStringList(str(raw.workspace_roots_json)),
    rawPath: str(raw.raw_path),
    sourceStamp: str(raw.source_stamp),
    discoveryState: normalizeDiscoveryState(raw.discovery_state),
    fromCache: raw.from_cache === true,
    ...(num(raw.parser_version) != null ? { parserVersion: num(raw.parser_version) as number } : {}),
  };
}

function diagnosticFromJson(raw: Record<string, unknown>): DiscoveryDiagnostic {
  return {
    source: str(raw.source) ?? '',
    locator: str(raw.locator),
    error: str(raw.error) ?? '',
  };
}

function providerSummaryFromJson(raw: unknown): ProviderDiscoverySummary {
  const obj = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;
  return {
    candidates: count(obj.candidates),
    discovered: count(obj.discovered),
    skippedUnchanged: count(obj.skipped_unchanged),
    failed: obj.failed === true,
  };
}

function summaryFromJson(raw: Record<string, unknown>): DiscoverySummary {
  const providersRaw = (typeof raw.providers === 'object' && raw.providers !== null
    ? raw.providers
    : {}) as Record<string, unknown>;
  const providers: Record<string, ProviderDiscoverySummary> = {};
  for (const [source, value] of Object.entries(providersRaw)) {
    providers[source] = providerSummaryFromJson(value);
  }
  const countersRaw = (typeof raw.counters === 'object' && raw.counters !== null
    ? raw.counters
    : {}) as Record<string, unknown>;
  return {
    contractVersion: count(raw.contract_version),
    discovered: count(raw.discovered),
    skippedUnchanged: count(raw.skipped_unchanged),
    providers,
    exemptSources: Array.isArray(raw.exempt_sources)
      ? raw.exempt_sources
          .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
          .map((item) => ({ source: str(item.source) ?? '', reason: str(item.reason) ?? '' }))
      : [],
    counters: {
      candidatesEnumerated: count(countersRaw.candidates_enumerated),
      shallowReads: count(countersRaw.shallow_reads),
      skippedUnchanged: count(countersRaw.skipped_unchanged),
      filesOpened: count(countersRaw.files_opened),
      bytesRead: count(countersRaw.bytes_read),
    },
  };
}

/**
 * Run shallow session discovery by driving `ai-hist sessions discover --json`.
 *
 * The CLI streams JSONL — one `session` object per discovered row, a
 * `diagnostic` per non-fatal failure, then a closing `summary` — and this
 * parses it as it arrives, so `onSession` fires progressively rather than only
 * once the whole run finishes.
 *
 * Discovery **writes the on-disk database**. An `AiHist` opened before the run
 * is a snapshot and will not contain the new rows; re-open (or call
 * `openAiHist` again) before `listSessionCatalog()` to see them.
 *
 * Failure behavior mirrors the CLI's: a provider that blows up contributes a
 * diagnostic and the run still resolves (exit 0). This rejects only when the
 * binary cannot be run at all, when every selected provider failed (non-zero
 * exit), or when the process errors for some other reason. Individual
 * unparseable lines are skipped rather than failing the run — a future binary
 * may add line types this SDK does not know.
 */
export function discoverSessions(opts: DiscoverSessionsOptions = {}): Promise<DiscoverResult> {
  const spawnFn = opts.spawnFn ?? spawn;
  const bin = resolveAiHistBinary(opts.binPath);

  const args = ['sessions', 'discover', '--json'];
  for (const source of opts.sources ?? []) args.push('--source', String(source));
  if (opts.limit != null) args.push('--limit', String(opts.limit));

  const spawnOpts: SpawnOptions = {
    env: { ...process.env, ...opts.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  };

  return new Promise((resolve, reject) => {
    const sessions: CatalogSession[] = [];
    const diagnostics: DiscoveryDiagnostic[] = [];
    let summary: DiscoverySummary | null = null;
    let stderr = '';
    let pending = '';
    const decoder = new StringDecoder('utf8');

    const handleLine = (line: string): void => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(trimmed);
      } catch {
        return; // Tolerant: skip a line we cannot read rather than failing the run.
      }
      if (typeof parsed !== 'object' || parsed === null) return;
      const obj = parsed as Record<string, unknown>;
      switch (obj.type) {
        case 'session': {
          const session = catalogSessionFromJson(obj);
          sessions.push(session);
          opts.onSession?.(session);
          break;
        }
        case 'diagnostic': {
          const diagnostic = diagnosticFromJson(obj);
          diagnostics.push(diagnostic);
          opts.onDiagnostic?.(diagnostic);
          break;
        }
        case 'summary':
          summary = summaryFromJson(obj);
          break;
        default:
          break; // Unknown line type from a newer binary.
      }
    };

    const child = spawnFn(bin, args, spawnOpts);

    child.stdout?.on('data', (chunk: Buffer | string) => {
      pending += typeof chunk === 'string' ? chunk : decoder.write(chunk);
      let newline = pending.indexOf('\n');
      while (newline !== -1) {
        handleLine(pending.slice(0, newline));
        pending = pending.slice(newline + 1);
        newline = pending.indexOf('\n');
      }
    });
    child.stderr?.on('data', (chunk) => {
      stderr += String(chunk);
    });

    child.on('error', (err: NodeJS.ErrnoException) => {
      if (MISSING_BINARY_CODES.has(err.code ?? '')) {
        reject(
          new Error(
            `could not run the ai-hist binary (${bin}): ${err.code}. Install ai-hist, ` +
              `or set ${AI_HIST_RUST_BIN_ENV} to its path.`,
          ),
        );
        return;
      }
      reject(err);
    });

    child.on('close', (code) => {
      handleLine(pending + decoder.end());
      pending = '';
      if (code !== 0) {
        reject(
          new Error(`ai-hist sessions discover failed (exit ${code}): ${stderr.trim().slice(0, 300)}`),
        );
        return;
      }
      resolve({ sessions, diagnostics, summary });
    });
  });
}
