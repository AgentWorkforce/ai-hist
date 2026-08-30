#!/usr/bin/env node

/** Thin MCP adapters over the public `ai-hist` TypeScript SDK. */
import { readFileSync } from 'node:fs';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import {
  discoverSessions, getSession, getSessionEventsPage, listSessionCatalogPage,
  recent, search, stats, sync,
} from './index.js';

const READ = { readOnlyHint: true, idempotentHint: true, openWorldHint: false } as const;
const WRITE = { readOnlyHint: false, idempotentHint: true, openWorldHint: false } as const;
const SOURCE = z.enum(['claude', 'codex', 'cursor', 'grok', 'relay', 'trajectory', 'opencode']);
const CATALOG_SOURCE = z.enum(['claude', 'codex', 'cursor', 'grok', 'relay', 'opencode']);
const packageVersion = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
).version as string;

const server = new McpServer(
  { name: 'ai-hist', version: packageVersion },
  { capabilities: { tools: {} } },
);

function result(value: unknown) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }] };
}

async function call(operation: () => Promise<unknown>) {
  try { return result(await operation()); }
  catch (error) {
    const value = error as { code?: string; message?: string };
    return { content: [{ type: 'text' as const, text: `${value.code ?? 'ERROR'}: ${value.message ?? String(error)}` }], isError: true };
  }
}

server.tool('search_history', 'Search already-indexed RelayHistory prompts.', {
  query: z.string(), source: SOURCE.optional(), project: z.string().optional(), tag: z.string().optional(),
  limit: z.number().int().min(1).max(1000).optional().default(20),
}, READ, ({ query, source, project, tag, limit }) => call(() => search(query, { source, project, tag, limit })));

server.tool('recent_history', 'List recent already-indexed history.', {
  source: SOURCE.optional(), project: z.string().optional(), tag: z.string().optional(),
  n: z.number().int().min(1).max(1000).optional().default(20),
  before_ms: z.number().int().optional(),
}, READ, ({ source, project, tag, n, before_ms }) => call(() => recent({ source, project, tag, limit: n, beforeMs: before_ms })));

server.tool('list_sessions', 'Cache-only indexed session catalog listing. This never discovers or syncs.', {
  sources: z.array(CATALOG_SOURCE).optional(), limit: z.number().int().min(1).max(1000).optional().default(20),
  before_ms: z.number().int().optional(),
  after: z.object({ lastActivityMs: z.number().int().nullable().optional(), source: z.string(), sessionId: z.string() }).optional(),
}, READ, ({ sources, limit, before_ms, after }) => call(() => listSessionCatalogPage({
  sources, limit, beforeMs: before_ms,
  after: after ? { ...after, lastActivityMs: after.lastActivityMs ?? null } : undefined,
})));

server.tool('discover_sessions', 'Explicit shallow provider discovery. Updates only the session catalog.', {
  sources: z.array(CATALOG_SOURCE).optional(), limit: z.number().int().min(1).max(10000).optional(),
}, WRITE, (args) => call(() => discoverSessions(args)));

server.tool('get_session', 'Get indexed prompts for one session.', {
  session_id: z.string(), source: SOURCE.optional(), tag: z.string().optional(),
}, READ, ({ session_id, source, tag }) => call(() => getSession(session_id, { source, tag })));

server.tool('get_session_events', 'Get one bounded page of normalized events.', {
  session_id: z.string(), source: SOURCE.optional(), limit: z.number().int().min(1).max(1000).optional().default(200),
  after: z.object({ tsMs: z.number().int(), id: z.number().int() }).optional(),
}, READ, ({ session_id, source, limit, after }) => call(() => getSessionEventsPage(session_id, { source, limit, after })));

server.tool('history_stats', 'Statistics for already-indexed RelayHistory data.', {
  tag: z.string().optional(),
}, READ, ({ tag }) => call(() => stats({ tag })));

server.tool('sync', 'Explicit full provider ingestion into RelayHistory.', {}, WRITE, () => call(() => sync()));

await server.connect(new StdioServerTransport());
