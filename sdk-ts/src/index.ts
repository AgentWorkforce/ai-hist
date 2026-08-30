/**
 * RelayHistory's public TypeScript API.
 *
 * Every production operation crosses one mandatory Node-API boundary into the
 * Rust engine. This module owns only input defaults, object normalization,
 * pagination ergonomics, and stable JavaScript errors.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

export const NATIVE_CONTRACT_VERSION = 2;
export const SESSION_CATALOG_CONTRACT_VERSION = 1;

export type Source = 'claude' | 'codex' | 'cursor' | 'grok' | 'relay' | 'trajectory' | 'opencode';
export type CatalogSource = Exclude<Source, 'trajectory'>;

export class RelayHistoryError extends Error {
  constructor(message: string, readonly code: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class UnsupportedPlatformError extends RelayHistoryError {}
export class NativePackageMissingError extends RelayHistoryError {}
export class NativeLoadError extends RelayHistoryError {}
export class NativeContractMismatchError extends RelayHistoryError {}
export class DatabaseOpenError extends RelayHistoryError {}
export class InvalidArgumentError extends RelayHistoryError {}

export interface HistoryEntry {
  id: number;
  source: Source;
  sessionId: string | null;
  project: string | null;
  prompt: string;
  timestampMs: number;
}

export interface ListOptions {
  dbPath?: string;
  source?: Source;
  project?: string;
  tag?: string;
  beforeMs?: number;
  limit?: number;
}

export interface SearchOptions extends ListOptions {
  rawFts?: boolean;
}

export interface SessionOptions {
  dbPath?: string;
  source?: Source;
  tag?: string;
}

export interface CatalogCursor {
  lastActivityMs: number | null;
  source: string;
  sessionId: string;
}

export interface CatalogSession {
  source: CatalogSource;
  sessionId: string;
  cwd: string | null;
  gitBranch: string | null;
  firstActivityMs: number | null;
  lastActivityMs: number | null;
  firstPrompt: string | null;
  lastAssistantText: string | null;
  models: string[];
  originator: string | null;
  agentVersion: string | null;
  repoUrl: string | null;
  initialCommit: string | null;
  workspaceRoots: string[];
  rawPath: string | null;
  sourceStamp: string | null;
  discoveryState: 'shallow' | 'full';
  fromCache: boolean;
}

export interface ListCatalogOptions {
  dbPath?: string;
  sources?: CatalogSource[];
  limit?: number;
  beforeMs?: number;
  after?: CatalogCursor;
}

export interface SessionCatalogPage {
  contractVersion: number;
  sessions: CatalogSession[];
  nextCursor: CatalogCursor | null;
}

export interface DiscoveryDiagnostic {
  source: string;
  locator: string | null;
  error: string;
}

export interface ProviderDiscoverySummary {
  source: string;
  candidates: number;
  discovered: number;
  skippedUnchanged: number;
  failed: boolean;
}

export interface DiscoveryCounters {
  candidatesEnumerated: number;
  shallowReads: number;
  skippedUnchanged: number;
  filesOpened: number;
  bytesRead: number;
}

export interface SourceExemption { source: string; reason: string }

export interface DiscoverSessionsOptions {
  dbPath?: string;
  sources?: CatalogSource[];
  limit?: number;
}

export interface DiscoverResult {
  contractVersion: number;
  sessions: CatalogSession[];
  discovered: number;
  skippedUnchanged: number;
  providers: ProviderDiscoverySummary[];
  exemptSources: SourceExemption[];
  diagnostics: DiscoveryDiagnostic[];
  counters: DiscoveryCounters;
}

export interface SessionEvent {
  id: number;
  source: Source;
  sessionId: string;
  project: string | null;
  cwd: string | null;
  gitBranch: string | null;
  messageId: string | null;
  parentId: string | null;
  tsMs: number;
  role: 'user' | 'assistant' | 'tool_result';
  kind: 'text' | 'thinking' | 'tool_use' | 'tool_result';
  text: string | null;
  model: string | null;
  tokenUsage: Record<string, unknown> | null;
  eventUid: string;
}

export interface EventCursor { tsMs: number; id: number }

export interface EventsPageOptions {
  dbPath?: string;
  source?: Source;
  limit?: number;
  after?: EventCursor;
}

export interface SessionEventsPage {
  events: SessionEvent[];
  nextCursor: EventCursor | null;
}

export interface Stats {
  total: number;
  bySource: Partial<Record<Source, number>>;
  byProject: Array<{ project: string; count: number }>;
  firstTimestampMs: number | null;
  lastTimestampMs: number | null;
}

export interface StatsOptions { dbPath?: string; tag?: string }
export interface SyncOptions { dbPath?: string }
export interface SyncResult { databasePath: string; completed: boolean }

type UnknownRecord = Record<string, unknown>;

interface NativeBinding {
  nativeContractVersion(): number;
  nativeBuildProfile?(): string;
  search(query: string, options?: object): Promise<UnknownRecord[]>;
  recent(options?: object): Promise<UnknownRecord[]>;
  getSession(sessionId: string, options?: object): Promise<UnknownRecord[]>;
  getSessionEventsPage(sessionId: string, options?: object): Promise<UnknownRecord>;
  stats(options?: object): Promise<UnknownRecord>;
  listSessionCatalog(options?: object): Promise<UnknownRecord[]>;
  listSessionCatalogPage(options?: object): Promise<UnknownRecord>;
  discoverSessions(options?: object): Promise<UnknownRecord>;
  sync(options?: object): Promise<UnknownRecord>;
}

const SUPPORTED_PLATFORMS = new Set([
  'darwin-arm64', 'darwin-x64',
  'linux-arm64-gnu', 'linux-arm64-musl',
  'linux-x64-gnu', 'linux-x64-musl',
  'win32-x64-msvc',
]);

function linuxLibc(): 'gnu' | 'musl' {
  const report = process.report?.getReport() as { header?: { glibcVersionRuntime?: string } } | undefined;
  return report?.header?.glibcVersionRuntime ? 'gnu' : 'musl';
}

export function runtimePlatform(): string {
  if (process.platform === 'linux') return `linux-${process.arch}-${linuxLibc()}`;
  if (process.platform === 'win32') return `win32-${process.arch}-msvc`;
  return `${process.platform}-${process.arch}`;
}

let nativePromise: Promise<NativeBinding> | null = null;

export function validateNativeContract(actual: number): void {
  if (actual !== NATIVE_CONTRACT_VERSION) {
    throw new NativeContractMismatchError(
      `ai-hist requires native contract ${NATIVE_CONTRACT_VERSION}, but ai-hist-native provides ${actual}. Reinstall matching versions.`,
      'NATIVE_CONTRACT_MISMATCH',
    );
  }
}

async function loadNative(): Promise<NativeBinding> {
  if (nativePromise) return nativePromise;
  nativePromise = (async () => {
    const platform = runtimePlatform();
    if (!SUPPORTED_PLATFORMS.has(platform)) {
      throw new UnsupportedPlatformError(
        `RelayHistory has no native build for ${platform}. Supported platforms: ${[...SUPPORTED_PLATFORMS].join(', ')}.`,
        'UNSUPPORTED_PLATFORM',
      );
    }
    let loaded: unknown;
    try {
      // Kept as a variable so TypeScript does not require native build-time
      // declarations; npm installs this mandatory production dependency.
      const packageName = 'ai-hist-native';
      loaded = await import(packageName);
    } catch (cause) {
      const error = cause as NodeJS.ErrnoException;
      if (error.code === 'ERR_MODULE_NOT_FOUND' || error.code === 'MODULE_NOT_FOUND') {
        throw new NativePackageMissingError(
          `RelayHistory supports ${platform}, but its native package is missing. Reinstall ai-hist with optional dependencies enabled.`,
          'NATIVE_PACKAGE_MISSING',
          { cause },
        );
      }
      throw new NativeLoadError(
        `RelayHistory's native package for ${platform} failed to load: ${error.message}`,
        'NATIVE_LOAD_FAILED',
        { cause },
      );
    }
    const binding = ((loaded as { default?: unknown }).default ?? loaded) as NativeBinding;
    if (typeof binding.nativeContractVersion !== 'function') {
      throw new NativeContractMismatchError(
        'The installed ai-hist-native package does not expose a contract version. Reinstall matching ai-hist packages.',
        'NATIVE_CONTRACT_MISMATCH',
      );
    }
    validateNativeContract(binding.nativeContractVersion());
    return binding;
  })();
  return nativePromise.catch((error) => {
    nativePromise = null;
    throw error;
  });
}

function nativeMessage(error: unknown): { code: string; message: string } | null {
  const message = error instanceof Error ? error.message : String(error);
  const match = message.match(/RELAYHISTORY_NATIVE::([A-Z_]+)::([\s\S]*)/);
  return match ? { code: match[1], message: match[2] } : null;
}

async function nativeCall<T>(call: (binding: NativeBinding) => Promise<T>): Promise<T> {
  try {
    return await call(await loadNative());
  } catch (error) {
    if (error instanceof RelayHistoryError) throw error;
    const native = nativeMessage(error);
    if (!native) throw new NativeLoadError(String(error), 'NATIVE_CALL_FAILED', { cause: error });
    if (native.code === 'DATABASE_OPEN_FAILED') {
      throw new DatabaseOpenError(native.message, native.code, { cause: error });
    }
    if (native.code === 'INVALID_ARGUMENT') {
      throw new InvalidArgumentError(native.message, native.code, { cause: error });
    }
    throw new RelayHistoryError(native.message, native.code, { cause: error });
  }
}

function nullableString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function historyEntry(value: UnknownRecord): HistoryEntry {
  return {
    id: Number(value.id),
    source: String(value.source) as Source,
    sessionId: nullableString(value.sessionId),
    project: nullableString(value.project),
    prompt: String(value.prompt),
    timestampMs: Number(value.timestampMs),
  };
}

function catalogCursor(value: unknown): CatalogCursor | null {
  if (!value || typeof value !== 'object') return null;
  const row = value as UnknownRecord;
  return {
    lastActivityMs: typeof row.lastActivityMs === 'number' ? row.lastActivityMs : null,
    source: String(row.source),
    sessionId: String(row.sessionId),
  };
}

function catalogSession(value: UnknownRecord): CatalogSession {
  return {
    source: String(value.source) as CatalogSource,
    sessionId: String(value.sessionId),
    cwd: nullableString(value.cwd),
    gitBranch: nullableString(value.gitBranch),
    firstActivityMs: typeof value.firstActivityMs === 'number' ? value.firstActivityMs : null,
    lastActivityMs: typeof value.lastActivityMs === 'number' ? value.lastActivityMs : null,
    firstPrompt: nullableString(value.firstPrompt),
    lastAssistantText: nullableString(value.lastAssistantText),
    models: Array.isArray(value.models) ? value.models.map(String) : [],
    originator: nullableString(value.originator),
    agentVersion: nullableString(value.agentVersion),
    repoUrl: nullableString(value.repoUrl),
    initialCommit: nullableString(value.initialCommit),
    workspaceRoots: Array.isArray(value.workspaceRoots) ? value.workspaceRoots.map(String) : [],
    rawPath: nullableString(value.rawPath),
    sourceStamp: nullableString(value.sourceStamp),
    discoveryState: value.discoveryState === 'shallow' ? 'shallow' : 'full',
    fromCache: value.fromCache === true,
  };
}

function assertCatalogContract(value: number): void {
  if (value !== SESSION_CATALOG_CONTRACT_VERSION) {
    throw new NativeContractMismatchError(
      `ai-hist expects catalog contract ${SESSION_CATALOG_CONTRACT_VERSION}, but native returned ${value}.`,
      'CATALOG_CONTRACT_MISMATCH',
    );
  }
}

function tokenUsage(raw: unknown): Record<string, unknown> | null {
  if (typeof raw !== 'string') return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function sessionEvent(value: UnknownRecord): SessionEvent {
  return {
    id: Number(value.id), source: String(value.source) as Source, sessionId: String(value.sessionId),
    project: nullableString(value.project), cwd: nullableString(value.cwd), gitBranch: nullableString(value.gitBranch),
    messageId: nullableString(value.messageId), parentId: nullableString(value.parentId), tsMs: Number(value.tsMs),
    role: String(value.role) as SessionEvent['role'], kind: String(value.kind) as SessionEvent['kind'],
    text: nullableString(value.text), model: nullableString(value.model), tokenUsage: tokenUsage(value.tokenJson),
    eventUid: String(value.eventUid),
  };
}

export function defaultDbPath(): string {
  if (process.env.AI_HIST_DB !== undefined) return process.env.AI_HIST_DB;
  if (process.env.XDG_DATA_HOME !== undefined) {
    return join(process.env.XDG_DATA_HOME, 'ai-hist', 'ai-history.db');
  }
  return join(homedir(), '.local', 'share', 'ai-hist', 'ai-history.db');
}

/**
 * Optimization profile of the loaded native addon: 'release', 'debug', or
 * 'unknown' for an addon predating the probe. Performance measurements are
 * only meaningful against 'release'.
 */
export async function nativeBuildProfile(): Promise<string> {
  return nativeCall(async (native) => native.nativeBuildProfile?.() ?? 'unknown');
}

export async function search(query: string, options: SearchOptions = {}): Promise<HistoryEntry[]> {
  return nativeCall(async (native) => (await native.search(query, options)).map(historyEntry));
}

export async function recent(options: ListOptions = {}): Promise<HistoryEntry[]> {
  return nativeCall(async (native) => (await native.recent(options)).map(historyEntry));
}

export async function getSession(sessionId: string, options: SessionOptions = {}): Promise<HistoryEntry[]> {
  return nativeCall(async (native) => (await native.getSession(sessionId, options)).map(historyEntry));
}

export async function listSessionCatalog(options: ListCatalogOptions = {}): Promise<CatalogSession[]> {
  return (await listSessionCatalogPage(options)).sessions;
}

export async function listSessionCatalogPage(options: ListCatalogOptions = {}): Promise<SessionCatalogPage> {
  return nativeCall(async (native) => {
    const page = await native.listSessionCatalogPage({ ...options, after: options.after ? {
      ...options.after,
      lastActivityMs: options.after.lastActivityMs ?? undefined,
    } : undefined });
    const contractVersion = Number(page.contractVersion);
    assertCatalogContract(contractVersion);
    return {
      contractVersion,
      sessions: Array.isArray(page.sessions) ? (page.sessions as UnknownRecord[]).map(catalogSession) : [],
      nextCursor: catalogCursor(page.nextCursor),
    };
  });
}

export async function discoverSessions(options: DiscoverSessionsOptions = {}): Promise<DiscoverResult> {
  return nativeCall(async (native) => {
    const result = await native.discoverSessions(options);
    const contractVersion = Number(result.contractVersion);
    assertCatalogContract(contractVersion);
    return {
      contractVersion,
      sessions: Array.isArray(result.sessions) ? (result.sessions as UnknownRecord[]).map(catalogSession) : [],
      discovered: Number(result.discovered),
      skippedUnchanged: Number(result.skippedUnchanged),
      providers: (result.providers as ProviderDiscoverySummary[]) ?? [],
      exemptSources: (result.exemptSources as SourceExemption[]) ?? [],
      diagnostics: Array.isArray(result.diagnostics) ? (result.diagnostics as UnknownRecord[]).map((item) => ({
        source: String(item.source), locator: nullableString(item.locator), error: String(item.error),
      })) : [],
      counters: result.counters as unknown as DiscoveryCounters,
    };
  });
}

export async function discoverAndList(options: DiscoverSessionsOptions & ListCatalogOptions = {}): Promise<SessionCatalogPage> {
  await discoverSessions(options);
  return listSessionCatalogPage(options);
}

export async function getSessionEventsPage(sessionId: string, options: EventsPageOptions = {}): Promise<SessionEventsPage> {
  return nativeCall(async (native) => {
    const page = await native.getSessionEventsPage(sessionId, options);
    return {
      events: Array.isArray(page.events) ? (page.events as UnknownRecord[]).map(sessionEvent) : [],
      nextCursor: page.nextCursor && typeof page.nextCursor === 'object'
        ? { tsMs: Number((page.nextCursor as UnknownRecord).tsMs), id: Number((page.nextCursor as UnknownRecord).id) }
        : null,
    };
  });
}

export async function* sessionEvents(sessionId: string, options: Omit<EventsPageOptions, 'after'> = {}): AsyncGenerator<SessionEvent> {
  let after: EventCursor | undefined;
  do {
    const page = await getSessionEventsPage(sessionId, { ...options, after });
    for (const event of page.events) yield event;
    after = page.nextCursor ?? undefined;
  } while (after);
}

export async function getSessionEvents(sessionId: string, options: Omit<EventsPageOptions, 'after'> = {}): Promise<SessionEvent[]> {
  const events: SessionEvent[] = [];
  for await (const event of sessionEvents(sessionId, options)) events.push(event);
  return events;
}

export async function stats(options: StatsOptions = {}): Promise<Stats> {
  return nativeCall(async (native) => {
    const result = await native.stats(options);
    const bySource: Partial<Record<Source, number>> = {};
    for (const item of (result.bySource as UnknownRecord[] | undefined) ?? []) {
      bySource[String(item.source) as Source] = Number(item.count);
    }
    return {
      total: Number(result.total), bySource,
      byProject: ((result.byProject as UnknownRecord[] | undefined) ?? []).map((item) => ({ project: String(item.project), count: Number(item.count) })),
      firstTimestampMs: typeof result.firstTimestampMs === 'number' ? result.firstTimestampMs : null,
      lastTimestampMs: typeof result.lastTimestampMs === 'number' ? result.lastTimestampMs : null,
    };
  });
}

export async function sync(options: SyncOptions = {}): Promise<SyncResult> {
  return nativeCall(async (native) => {
    const result = await native.sync(options);
    return { databasePath: String(result.databasePath), completed: result.completed === true };
  });
}

export function resumeCommand(entry: Pick<HistoryEntry, 'source' | 'sessionId' | 'project'>): string | null {
  if (!entry.sessionId) return null;
  const resume = (() => {
    if (entry.source === 'claude') return `claude --resume ${shellQuote(entry.sessionId)}`;
    if (entry.source === 'codex') return `codex resume ${shellQuote(entry.sessionId)}`;
    if (entry.source === 'cursor') return `cursor-agent --resume=${shellQuote(entry.sessionId)}`;
    if (entry.source === 'grok') return `grok resume ${shellQuote(entry.sessionId)}`;
    return null;
  })();
  return resume && entry.project ? `cd ${shellQuote(entry.project)} && ${resume}` : resume;
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9._:/-]+$/.test(value) ? value : `'${value.replace(/'/g, `'"'"'`)}'`;
}
