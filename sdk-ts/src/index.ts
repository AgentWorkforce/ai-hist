/**
 * RelayHistory's public TypeScript API.
 *
 * Every production operation crosses one mandatory Node-API boundary into the
 * Rust engine. This module owns only input defaults, object normalization,
 * pagination ergonomics, and stable JavaScript errors.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

export const NATIVE_CONTRACT_VERSION = 5;
export const SESSION_CATALOG_CONTRACT_VERSION = 2;
export const SESSION_HYDRATION_CONTRACT_VERSION = 1;
export const SESSION_RELATIONSHIP_CONTRACT_VERSION = 1;

export type Source = 'claude' | 'codex' | 'cursor' | 'grok' | 'relay' | 'trajectory' | 'opencode';
export type CatalogSource = Exclude<Source, 'trajectory'>;
export type SessionScope = 'local' | 'remote' | 'all';
export type SessionLocation = Exclude<SessionScope, 'all'>;

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
export class UnsupportedOperationError extends RelayHistoryError {}
export class SessionNotFoundError extends RelayHistoryError {}
export class SessionSourceUnavailableError extends RelayHistoryError {}
export class SessionSourceMismatchError extends RelayHistoryError {}
export class HydrationUnsupportedError extends RelayHistoryError {}
export class HydrationFailedError extends RelayHistoryError {}

export interface HistoryEntry {
  id: number;
  source: Source;
  sessionId: string | null;
  project: string | null;
  prompt: string;
  timestampMs: number;
  locations: SessionLocation[];
}

export interface ListOptions {
  dbPath?: string;
  scope?: SessionScope;
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
  locations: SessionLocation[];
}

export interface ListCatalogOptions {
  dbPath?: string;
  scope?: SessionScope;
  sources?: CatalogSource[];
  limit?: number;
  beforeMs?: number;
  after?: CatalogCursor;
}

export interface SessionCatalogPage {
  contractVersion: number;
  scope: SessionScope;
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
  scope?: SessionScope;
  sources?: CatalogSource[];
  limit?: number;
}

export interface DiscoverResult {
  contractVersion: number;
  scope: SessionScope;
  /** Connector locations that actually executed. `scope` records the ask;
   * this records what ran — an `all` request executes remote connectors only
   * where one is configured on this machine. */
  locationsRun: SessionLocation[];
  sessions: CatalogSession[];
  discovered: number;
  skippedUnchanged: number;
  providers: ProviderDiscoverySummary[];
  exemptSources: SourceExemption[];
  diagnostics: DiscoveryDiagnostic[];
  counters: DiscoveryCounters;
}

export interface SessionRef {
  source: CatalogSource;
  sessionId: string;
  scope?: SessionScope;
}

export interface HydrateSessionOptions extends SessionRef {
  dbPath?: string;
  includeRelated?: boolean;
}

export interface HydrationDiagnostic {
  code: string;
  message: string;
  durationMs: number | null;
  sourceBytes: number | null;
  recordsParsed: number | null;
}

export interface HydrateSessionResult {
  contractVersion: number;
  source: CatalogSource;
  sessionId: string;
  status: 'hydrated' | 'updated' | 'unchanged';
  discoveryState: 'full';
  presence: SessionLocation;
  indexedThrough: {
    sourceStamp: string | null;
    lastEventAtMs: number | null;
  };
  evidence: {
    prompts: number;
    events: number;
    toolCalls: number;
    relatedSessions: number;
  };
  relatedSessionIds: string[];
  diagnostics: HydrationDiagnostic[];
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

export type RelationshipType = 'delegated';
export type IdentityStatus = 'observed' | 'unlinked';
export type StableChildIdentity = 'always' | 'sometimes' | 'never';

/**
 * One observed delegation edge. `childSessionId` is null when the provider
 * recorded the delegation but no stable child identity, in which case
 * `identityStatus` is `unlinked` and the child's output stays attributed to
 * the parent.
 */
export interface SessionRelationship {
  source: CatalogSource;
  parentSessionId: string;
  childSessionId: string | null;
  relationship: RelationshipType;
  identityStatus: IdentityStatus;
  childAgentType: string | null;
  childAgentName: string | null;
  childModel: string | null;
  spawnDepth: number | null;
  evidenceKind: string;
  evidenceLocator: string | null;
  evidenceRef: string | null;
  childHasEvents: boolean;
  spawnedAtMs: number | null;
  createdMs: number;
  relationshipUid: string;
}

/** What a provider is able to record about its own delegations. */
export interface RelationshipCapabilities {
  source: CatalogSource;
  stableChildIdentity: StableChildIdentity;
  recordsAgentType: boolean;
  recordsSpawnTime: boolean;
  recordsEvidenceLocator: boolean;
}

export interface RelationshipDiagnostic {
  code: string;
  message: string;
  relationshipUid: string | null;
}

export interface GetSessionRelationshipsOptions {
  source: CatalogSource;
  sessionId: string;
  dbPath?: string;
}

export interface SessionRelationships {
  contractVersion: number;
  source: CatalogSource;
  sessionId: string;
  /** Edges where this session is the delegating parent. */
  asParent: SessionRelationship[];
  /** Edges where this session is the delegated child. */
  asChild: SessionRelationship[];
  capabilities: RelationshipCapabilities;
  diagnostics: RelationshipDiagnostic[];
}

export interface SessionTreeNode {
  source: CatalogSource;
  sessionId: string;
  depth: number;
  parentSessionId: string | null;
  /** The edge that reached this node; null for the root. */
  relationship: SessionRelationship | null;
  childCount: number;
  hasEvents: boolean;
  /** Children exist but were not expanded (depth/node budget, or a cycle). */
  truncated: boolean;
}

export interface GetSessionTreeOptions extends GetSessionRelationshipsOptions {
  /** Default 32, maximum 64. */
  maxDepth?: number;
  /** Default 1000, maximum 10000. */
  maxNodes?: number;
}

export interface SessionTree {
  contractVersion: number;
  source: CatalogSource;
  rootSessionId: string;
  /** Pre-order and deterministic; `nodes[0]` is the root when it exists. */
  nodes: SessionTreeNode[];
  /** Related evidence at any depth with no stable child identity. */
  unlinked: SessionRelationship[];
  capabilities: RelationshipCapabilities;
  diagnostics: RelationshipDiagnostic[];
  truncated: boolean;
  maxDepthReached: number;
}

export interface RelationshipCursor {
  spawnedAtMs: number | null;
  relationshipUid: string;
}

export interface SessionChildrenPage {
  children: SessionRelationship[];
  nextCursor: RelationshipCursor | null;
}

export interface GetSessionChildrenPageOptions extends GetSessionRelationshipsOptions {
  /** Default 100, maximum 1000. */
  limit?: number;
  after?: RelationshipCursor;
}

export interface SessionDescendantsOptions extends GetSessionRelationshipsOptions {
  maxDepth?: number;
  pageLimit?: number;
}

export interface DescendantEventsOptions {
  source: CatalogSource;
  dbPath?: string;
  limit?: number;
  maxDepth?: number;
  /** Defaults to true. */
  includeRoot?: boolean;
}

export interface Stats {
  scope: SessionScope;
  total: number;
  bySource: Partial<Record<Source, number>>;
  byProject: Array<{ project: string; count: number }>;
  firstTimestampMs: number | null;
  lastTimestampMs: number | null;
}

export interface StatsOptions { dbPath?: string; scope?: SessionScope; tag?: string }
export interface SyncOptions { dbPath?: string; scope?: SessionScope }
export interface SyncResult { databasePath: string; scope: SessionScope; completed: boolean }

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
  hydrateSession(options: object): Promise<UnknownRecord>;
  getSessionRelationships(options: object): Promise<UnknownRecord>;
  getSessionTree(options: object): Promise<UnknownRecord>;
  getSessionChildrenPage(options: object): Promise<UnknownRecord>;
  sync(options?: object): Promise<UnknownRecord>;
}

const CATALOG_SOURCES: readonly string[] = ['claude', 'codex', 'cursor', 'grok', 'relay', 'opencode'];
const RELATIONSHIP_TYPES: readonly string[] = ['delegated'];
const DEFAULT_TREE_MAX_DEPTH = 32;
const MAX_TREE_MAX_DEPTH = 64;

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
    if (native.code === 'UNSUPPORTED_OPERATION') {
      throw new UnsupportedOperationError(native.message, native.code, { cause: error });
    }
    if (native.code === 'SESSION_NOT_FOUND') {
      throw new SessionNotFoundError(native.message, native.code, { cause: error });
    }
    if (native.code === 'SESSION_SOURCE_UNAVAILABLE') {
      throw new SessionSourceUnavailableError(native.message, native.code, { cause: error });
    }
    if (native.code === 'SESSION_SOURCE_MISMATCH') {
      throw new SessionSourceMismatchError(native.message, native.code, { cause: error });
    }
    if (native.code === 'HYDRATION_UNSUPPORTED') {
      throw new HydrationUnsupportedError(native.message, native.code, { cause: error });
    }
    if (native.code === 'HYDRATION_FAILED') {
      throw new HydrationFailedError(native.message, native.code, { cause: error });
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
    locations: Array.isArray(value.locations)
      ? value.locations.filter((location): location is SessionLocation => location === 'local' || location === 'remote')
      : [],
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
    locations: Array.isArray(value.locations)
      ? value.locations.filter((location): location is SessionLocation => location === 'local' || location === 'remote')
      : [],
  };
}

export function validateNativeLocation(value: unknown): SessionLocation {
  if (value === 'local' || value === 'remote') return value;
  throw new NativeContractMismatchError(
    `ai-hist-native returned an invalid session location: ${JSON.stringify(value)}. Reinstall matching ai-hist packages.`,
    'NATIVE_CONTRACT_MISMATCH',
  );
}

export function validateNativeScope(value: unknown): SessionScope {
  if (value === 'local' || value === 'remote' || value === 'all') return value;
  throw new NativeContractMismatchError(
    `ai-hist-native returned an invalid session scope: ${JSON.stringify(value)}. Reinstall matching ai-hist packages.`,
    'NATIVE_CONTRACT_MISMATCH',
  );
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

function assertRelationshipContract(value: number): void {
  if (value !== SESSION_RELATIONSHIP_CONTRACT_VERSION) {
    throw new NativeContractMismatchError(
      `ai-hist expects relationship contract ${SESSION_RELATIONSHIP_CONTRACT_VERSION}, but native returned ${value}.`,
      'RELATIONSHIP_CONTRACT_MISMATCH',
    );
  }
}

function catalogSource(value: unknown): CatalogSource {
  if (typeof value === 'string' && CATALOG_SOURCES.includes(value)) return value as CatalogSource;
  throw new NativeContractMismatchError(
    `ai-hist-native returned an invalid catalog source: ${JSON.stringify(value)}. Reinstall matching ai-hist packages.`,
    'NATIVE_CONTRACT_MISMATCH',
  );
}

function relationshipType(value: unknown): RelationshipType {
  if (typeof value === 'string' && RELATIONSHIP_TYPES.includes(value)) return value as RelationshipType;
  throw new NativeContractMismatchError(
    `ai-hist-native returned an invalid relationship type: ${JSON.stringify(value)}. Reinstall matching ai-hist packages.`,
    'NATIVE_CONTRACT_MISMATCH',
  );
}

function identityStatus(value: unknown): IdentityStatus {
  if (value === 'observed' || value === 'unlinked') return value;
  throw new NativeContractMismatchError(
    `ai-hist-native returned an invalid relationship identity status: ${JSON.stringify(value)}. Reinstall matching ai-hist packages.`,
    'NATIVE_CONTRACT_MISMATCH',
  );
}

function stableChildIdentity(value: unknown): StableChildIdentity {
  if (value === 'always' || value === 'sometimes' || value === 'never') return value;
  throw new NativeContractMismatchError(
    `ai-hist-native returned an invalid stable child identity: ${JSON.stringify(value)}. Reinstall matching ai-hist packages.`,
    'NATIVE_CONTRACT_MISMATCH',
  );
}

function relationship(value: UnknownRecord): SessionRelationship {
  return {
    source: catalogSource(value.source),
    parentSessionId: String(value.parentSessionId),
    childSessionId: nullableString(value.childSessionId),
    relationship: relationshipType(value.relationship),
    identityStatus: identityStatus(value.identityStatus),
    childAgentType: nullableString(value.childAgentType),
    childAgentName: nullableString(value.childAgentName),
    childModel: nullableString(value.childModel),
    spawnDepth: typeof value.spawnDepth === 'number' ? value.spawnDepth : null,
    evidenceKind: String(value.evidenceKind),
    evidenceLocator: nullableString(value.evidenceLocator),
    evidenceRef: nullableString(value.evidenceRef),
    childHasEvents: value.childHasEvents === true,
    spawnedAtMs: typeof value.spawnedAtMs === 'number' ? value.spawnedAtMs : null,
    createdMs: Number(value.createdMs),
    relationshipUid: String(value.relationshipUid),
  };
}

function relationships(value: unknown): SessionRelationship[] {
  return Array.isArray(value) ? (value as UnknownRecord[]).map(relationship) : [];
}

function relationshipCapabilities(value: unknown): RelationshipCapabilities {
  const row = (value ?? {}) as UnknownRecord;
  return {
    source: catalogSource(row.source),
    stableChildIdentity: stableChildIdentity(row.stableChildIdentity),
    recordsAgentType: row.recordsAgentType === true,
    recordsSpawnTime: row.recordsSpawnTime === true,
    recordsEvidenceLocator: row.recordsEvidenceLocator === true,
  };
}

function relationshipDiagnostics(value: unknown): RelationshipDiagnostic[] {
  return Array.isArray(value) ? (value as UnknownRecord[]).map((item) => ({
    code: String(item.code),
    message: String(item.message),
    relationshipUid: nullableString(item.relationshipUid),
  })) : [];
}

function treeNode(value: UnknownRecord): SessionTreeNode {
  return {
    source: catalogSource(value.source),
    sessionId: String(value.sessionId),
    depth: Number(value.depth),
    parentSessionId: nullableString(value.parentSessionId),
    relationship: value.relationship && typeof value.relationship === 'object'
      ? relationship(value.relationship as UnknownRecord)
      : null,
    childCount: Number(value.childCount),
    hasEvents: value.hasEvents === true,
    truncated: value.truncated === true,
  };
}

function relationshipCursor(value: unknown): RelationshipCursor | null {
  if (!value || typeof value !== 'object') return null;
  const row = value as UnknownRecord;
  return {
    spawnedAtMs: typeof row.spawnedAtMs === 'number' ? row.spawnedAtMs : null,
    relationshipUid: String(row.relationshipUid),
  };
}

function validateSessionRef(options: GetSessionRelationshipsOptions, operation: string): void {
  if (!options || typeof options !== 'object') {
    throw new InvalidArgumentError(`${operation} options are required`, 'INVALID_ARGUMENT');
  }
  if (!CATALOG_SOURCES.includes(options.source)) {
    throw new InvalidArgumentError(`invalid catalog source: ${String(options.source)}`, 'INVALID_ARGUMENT');
  }
  if (typeof options.sessionId !== 'string' || options.sessionId.trim() === '') {
    throw new InvalidArgumentError('sessionId must not be empty', 'INVALID_ARGUMENT');
  }
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
  return nativeCall(async (native) => (await native.search(query, { ...options, scope: options.scope ?? 'local' })).map(historyEntry));
}

export async function recent(options: ListOptions = {}): Promise<HistoryEntry[]> {
  return nativeCall(async (native) => (await native.recent({ ...options, scope: options.scope ?? 'local' })).map(historyEntry));
}

export async function getSession(sessionId: string, options: SessionOptions = {}): Promise<HistoryEntry[]> {
  return nativeCall(async (native) => (await native.getSession(sessionId, options)).map(historyEntry));
}

export async function listSessionCatalog(options: ListCatalogOptions = {}): Promise<CatalogSession[]> {
  return (await listSessionCatalogPage(options)).sessions;
}

export async function listSessionCatalogPage(options: ListCatalogOptions = {}): Promise<SessionCatalogPage> {
  return nativeCall(async (native) => {
    const page = await native.listSessionCatalogPage({ ...options, scope: options.scope ?? 'local', after: options.after ? {
      ...options.after,
      lastActivityMs: options.after.lastActivityMs ?? undefined,
    } : undefined });
    const contractVersion = Number(page.contractVersion);
    assertCatalogContract(contractVersion);
    return {
      contractVersion,
      scope: validateNativeScope(page.scope),
      sessions: Array.isArray(page.sessions) ? (page.sessions as UnknownRecord[]).map(catalogSession) : [],
      nextCursor: catalogCursor(page.nextCursor),
    };
  });
}

export async function discoverSessions(options: DiscoverSessionsOptions = {}): Promise<DiscoverResult> {
  return nativeCall(async (native) => {
    const result = await native.discoverSessions({ ...options, scope: options.scope ?? 'local' });
    const contractVersion = Number(result.contractVersion);
    assertCatalogContract(contractVersion);
    return {
      contractVersion,
      scope: validateNativeScope(result.scope),
      locationsRun: Array.isArray(result.locationsRun)
        ? (result.locationsRun as unknown[]).map(validateNativeLocation)
        : [],
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

export async function hydrateSession(options: HydrateSessionOptions): Promise<HydrateSessionResult> {
  if (!options || typeof options !== 'object') {
    throw new InvalidArgumentError('hydrateSession options are required', 'INVALID_ARGUMENT');
  }
  if (!CATALOG_SOURCES.includes(options.source)) {
    throw new InvalidArgumentError(`invalid catalog source: ${String(options.source)}`, 'INVALID_ARGUMENT');
  }
  if (typeof options.sessionId !== 'string' || options.sessionId.trim() === '') {
    throw new InvalidArgumentError('sessionId must not be empty', 'INVALID_ARGUMENT');
  }
  return nativeCall(async (native) => {
    const value = await native.hydrateSession({
      ...options,
      scope: options.scope ?? 'local',
      includeRelated: options.includeRelated ?? true,
    });
    const contractVersion = Number(value.contractVersion);
    if (contractVersion !== SESSION_HYDRATION_CONTRACT_VERSION) {
      throw new NativeContractMismatchError(
        `ai-hist expects hydration contract ${SESSION_HYDRATION_CONTRACT_VERSION}, but native returned ${contractVersion}.`,
        'NATIVE_CONTRACT_MISMATCH',
      );
    }
    if (!['hydrated', 'updated', 'unchanged'].includes(String(value.status)) || value.discoveryState !== 'full') {
      throw new NativeContractMismatchError(
        'ai-hist-native returned an invalid hydration result.',
        'NATIVE_CONTRACT_MISMATCH',
      );
    }
    const indexed = (value.indexedThrough ?? {}) as UnknownRecord;
    const evidence = (value.evidence ?? {}) as UnknownRecord;
    return {
      contractVersion,
      source: String(value.source) as CatalogSource,
      sessionId: String(value.sessionId),
      status: String(value.status) as HydrateSessionResult['status'],
      discoveryState: 'full',
      presence: value.presence === 'remote' ? 'remote' : 'local',
      indexedThrough: {
        sourceStamp: nullableString(indexed.sourceStamp),
        lastEventAtMs: typeof indexed.lastEventAtMs === 'number' ? indexed.lastEventAtMs : null,
      },
      evidence: {
        prompts: Number(evidence.prompts),
        events: Number(evidence.events),
        toolCalls: Number(evidence.toolCalls),
        relatedSessions: Number(evidence.relatedSessions),
      },
      relatedSessionIds: Array.isArray(value.relatedSessionIds) ? value.relatedSessionIds.map(String) : [],
      diagnostics: Array.isArray(value.diagnostics) ? (value.diagnostics as UnknownRecord[]).map((item) => ({
        code: String(item.code),
        message: String(item.message),
        durationMs: typeof item.durationMs === 'number' ? item.durationMs : null,
        sourceBytes: typeof item.sourceBytes === 'number' ? item.sourceBytes : null,
        recordsParsed: typeof item.recordsParsed === 'number' ? item.recordsParsed : null,
      })) : [],
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

/**
 * Direct delegation relationships for one session, in both directions. A
 * missing database returns an empty, well-formed result whose `capabilities`
 * still describe what the provider is able to record.
 */
export async function getSessionRelationships(options: GetSessionRelationshipsOptions): Promise<SessionRelationships> {
  validateSessionRef(options, 'getSessionRelationships');
  return nativeCall(async (native) => {
    const value = await native.getSessionRelationships({
      source: options.source, sessionId: options.sessionId, dbPath: options.dbPath,
    });
    const contractVersion = Number(value.contractVersion);
    assertRelationshipContract(contractVersion);
    return {
      contractVersion,
      source: String(value.source) as CatalogSource,
      sessionId: String(value.sessionId),
      asParent: relationships(value.asParent),
      asChild: relationships(value.asChild),
      capabilities: relationshipCapabilities(value.capabilities),
      diagnostics: relationshipDiagnostics(value.diagnostics),
    };
  });
}

/**
 * The complete descendant delegation tree for one session: pre-order,
 * cycle-safe, and bounded by `maxDepth` and `maxNodes`. Child events keep
 * their own session identity and are never flattened into the root.
 */
export async function getSessionTree(options: GetSessionTreeOptions): Promise<SessionTree> {
  validateSessionRef(options, 'getSessionTree');
  return nativeCall(async (native) => {
    const value = await native.getSessionTree({
      source: options.source, sessionId: options.sessionId, dbPath: options.dbPath,
      maxDepth: options.maxDepth, maxNodes: options.maxNodes,
    });
    const contractVersion = Number(value.contractVersion);
    assertRelationshipContract(contractVersion);
    return {
      contractVersion,
      source: String(value.source) as CatalogSource,
      rootSessionId: String(value.rootSessionId),
      nodes: Array.isArray(value.nodes) ? (value.nodes as UnknownRecord[]).map(treeNode) : [],
      unlinked: relationships(value.unlinked),
      capabilities: relationshipCapabilities(value.capabilities),
      diagnostics: relationshipDiagnostics(value.diagnostics),
      truncated: value.truncated === true,
      maxDepthReached: Number(value.maxDepthReached),
    };
  });
}

/**
 * One bounded page of a session's direct children, in the same total order
 * the tree traversal uses: `(spawnedAtMs, relationshipUid)`, nulls last.
 */
export async function getSessionChildrenPage(options: GetSessionChildrenPageOptions): Promise<SessionChildrenPage> {
  validateSessionRef(options, 'getSessionChildrenPage');
  return nativeCall(async (native) => {
    const page = await native.getSessionChildrenPage({
      source: options.source, sessionId: options.sessionId, dbPath: options.dbPath, limit: options.limit,
      after: options.after ? {
        ...options.after,
        spawnedAtMs: options.after.spawnedAtMs ?? undefined,
      } : undefined,
    });
    return {
      children: relationships(page.children),
      nextCursor: relationshipCursor(page.nextCursor),
    };
  });
}

/**
 * Lazily walks a session's descendants breadth-first over the paged children
 * primitive, without materializing a large tree. Unlinked evidence has no
 * traversable identity and is skipped; use `getSessionTree` when you need it.
 */
export async function* sessionDescendants(options: SessionDescendantsOptions): AsyncGenerator<SessionTreeNode> {
  validateSessionRef(options, 'sessionDescendants');
  const maxDepth = Math.min(Math.max(Math.trunc(options.maxDepth ?? DEFAULT_TREE_MAX_DEPTH), 1), MAX_TREE_MAX_DEPTH);
  const request = { source: options.source, dbPath: options.dbPath };
  const visited = new Set<string>([options.sessionId]);
  // Each walked session's parent, so a repeated edge can be told apart: back
  // into this branch's own ancestry it is a cycle, anywhere else it is a
  // diamond, which leaves nothing unexplored. `getSessionTree` draws the same
  // line, and the two must not disagree about what `truncated` means.
  const parentOf = new Map<string, string | null>([[options.sessionId, null]]);
  const isAncestor = (from: string, candidate: string): boolean => {
    let current: string | null | undefined = from;
    while (current !== null && current !== undefined) {
      if (current === candidate) return true;
      current = parentOf.get(current) ?? null;
    }
    return false;
  };
  let frontier: SessionTreeNode[] = [{
    source: options.source, sessionId: options.sessionId, depth: 0, parentSessionId: null,
    relationship: null, childCount: 0, hasEvents: false, truncated: false,
  }];
  while (frontier.length > 0) {
    const next: SessionTreeNode[] = [];
    for (const node of frontier) {
      const expand = node.depth < maxDepth;
      const children: SessionRelationship[] = [];
      let after: RelationshipCursor | undefined;
      do {
        const page = await getSessionChildrenPage({
          ...request, sessionId: node.sessionId, limit: options.pageLimit, after,
        });
        children.push(...page.children);
        after = page.nextCursor ?? undefined;
      } while (after);
      // Unlinked evidence has no traversable identity, so it is never a child
      // this walk owes the caller — at the depth boundary included.
      const linked = children.filter((edge): edge is SessionRelationship & { childSessionId: string } =>
        edge.identityStatus === 'observed' && edge.childSessionId !== null);
      node.childCount = linked.length;
      node.truncated = expand
        ? linked.some((edge) => isAncestor(node.sessionId, edge.childSessionId))
        : linked.length > 0;
      if (node.depth > 0) yield node;
      if (!expand) continue;
      for (const edge of linked) {
        if (visited.has(edge.childSessionId)) continue;
        visited.add(edge.childSessionId);
        parentOf.set(edge.childSessionId, node.sessionId);
        next.push({
          source: edge.source, sessionId: edge.childSessionId, depth: node.depth + 1,
          parentSessionId: node.sessionId, relationship: edge, childCount: 0,
          hasEvents: edge.childHasEvents, truncated: false,
        });
      }
    }
    frontier = next;
  }
}

/**
 * The root session's events followed by each descendant's events, in
 * descendant traversal order. Every yielded event keeps the `sessionId` of the
 * session that actually produced it: a child's event is never rewritten as a
 * parent's.
 */
export async function* sessionEventsIncludingDescendants(
  options: DescendantEventsOptions & { sessionId: string },
): AsyncGenerator<SessionEvent> {
  validateSessionRef(options, 'sessionEventsIncludingDescendants');
  const events = { dbPath: options.dbPath, source: options.source, limit: options.limit };
  if (options.includeRoot !== false) {
    yield* sessionEvents(options.sessionId, events);
  }
  for await (const node of sessionDescendants({
    source: options.source, sessionId: options.sessionId, dbPath: options.dbPath, maxDepth: options.maxDepth,
  })) {
    if (!node.hasEvents) continue;
    yield* sessionEvents(node.sessionId, events);
  }
}

export async function stats(options: StatsOptions = {}): Promise<Stats> {
  return nativeCall(async (native) => {
    const result = await native.stats({ ...options, scope: options.scope ?? 'local' });
    const bySource: Partial<Record<Source, number>> = {};
    for (const item of (result.bySource as UnknownRecord[] | undefined) ?? []) {
      bySource[String(item.source) as Source] = Number(item.count);
    }
    return {
      scope: validateNativeScope(result.scope),
      total: Number(result.total), bySource,
      byProject: ((result.byProject as UnknownRecord[] | undefined) ?? []).map((item) => ({ project: String(item.project), count: Number(item.count) })),
      firstTimestampMs: typeof result.firstTimestampMs === 'number' ? result.firstTimestampMs : null,
      lastTimestampMs: typeof result.lastTimestampMs === 'number' ? result.lastTimestampMs : null,
    };
  });
}

export async function sync(options: SyncOptions = {}): Promise<SyncResult> {
  return nativeCall(async (native) => {
    const result = await native.sync({ ...options, scope: options.scope ?? 'local' });
    return {
      databasePath: String(result.databasePath),
      scope: validateNativeScope(result.scope),
      completed: result.completed === true,
    };
  });
}

export function resumeCommand(entry: Pick<HistoryEntry, 'source' | 'sessionId' | 'project' | 'locations'>): string | null {
  if (!entry.sessionId) return null;
  if (entry.locations.length > 0 && !entry.locations.includes('local')) return null;
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
