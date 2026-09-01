# ai-hist

The public TypeScript SDK, Node CLI, and MCP server for RelayHistory. Every
operation uses the mandatory `ai-hist-native` Node-API engine; there is no
JavaScript SQLite implementation or provider-file fallback.

```bash
npm install ai-hist
```

```ts
import {
  discoverSessions,
  hydrateSession,
  listSessionCatalogPage,
  getSession,
  getSessionEventsPage,
  sessionEvents,
  search,
  recent,
  stats,
  sync,
} from 'ai-hist';

await discoverSessions({ scope: 'all', limit: 100 });
const page = await listSessionCatalogPage({ scope: 'all', limit: 20 });
const first = page.sessions[0];
if (first) {
  await hydrateSession({ source: first.source, sessionId: first.sessionId });
  const prompts = await getSession(first.sessionId);
  const events = await getSessionEventsPage(first.sessionId, { limit: 200 });
  for await (const event of sessionEvents(first.sessionId)) consume(event);
}
```

All APIs are async. `listSessionCatalog` and `listSessionCatalogPage` are
cache-only. `discoverSessions` is shallow discovery. `hydrateSession` is
targeted evidence acquisition for one existing catalog row. It returns
`hydrated`, `updated`, or `unchanged`, an indexed source stamp, evidence counts,
related native session IDs, and bounded-work metrics. `sync` is full ingestion.
Missing databases return empty read results; they do not trigger provider I/O.

Session discovery, listing, search, recent history, statistics, and sync accept
`scope: 'local' | 'remote' | 'all'`. Scope defaults to `local`, preserving
offline behavior and making provider-cloud access explicit. The CLI exposes the
same mutually exclusive `--local`, `--remote`, and `--all` flags; omitting them
is equivalent to `--local`.

Cached reads already support every scope. Remote acquisition runs through
provider connectors — claude.ai/code web sessions and Codex cloud tasks —
that are configured by the provider CLI's own sign-in on the machine (see the
repository's `docs/remote-connectors.md`). On a machine with no connector
configured, `discoverSessions({ scope: 'remote' })` and
`sync({ scope: 'remote' })` fail with `UnsupportedOperationError` and the stable
code `UNSUPPORTED_OPERATION`; they never fall back to local acquisition.
`scope: 'all'` runs the local adapters plus every configured connector.

Catalog pages, discovery results, statistics, and sync results echo the requested `scope`,
and discovery results additionally report `locationsRun` — the connector
locations that actually executed.
History and catalog rows have `locations`, containing `local`, `remote`, or both, so an
`all` query still returns one logical session while preserving where it was
found. `resumeCommand()` returns `null` for a remote-only history row rather
than emitting a local CLI command; an empty `locations` array retains legacy
local behavior for rows written before provenance tracking.

The event primitive is page-based and uses `{ tsMs, id }` as a deterministic
cursor. The `sessionEvents` async iterator walks pages without accumulating a
large transcript. `getSessionEvents` is an explicit collecting convenience.

## Delegation topology

Sessions that delegate to subagents form a tree, and it is queryable:

```ts
import {
  getSessionRelationships,
  getSessionTree,
  sessionEventsIncludingDescendants,
} from 'ai-hist';

const { asParent, asChild, capabilities } = await getSessionRelationships({
  source: 'codex',
  sessionId: rootId,
});
const tree = await getSessionTree({ source: 'codex', sessionId: rootId, maxDepth: 8 });

for await (const event of sessionEventsIncludingDescendants({ source: 'codex', sessionId: rootId })) {
  // event.sessionId is the session that actually produced the event: a
  // child's event is never rewritten as the parent's.
  consume(event);
}
```

Each `SessionRelationship` reports its `identityStatus`. `observed` means the
provider named the child, so `childSessionId` is a real session id and
`childHasEvents` says whether its events are independently addressable through
`getSessionEventsPage`. `unlinked` means the provider recorded a delegation but
no stable child identity: `childSessionId` is `null`, the child's output stays
attributed to the parent, and the row keeps the evidence (`evidenceKind`,
`evidenceLocator`, `evidenceRef`) that established it. An identity is never
synthesized.

`capabilities.stableChildIdentity` tells you what to expect from the provider
before you read a single row: `always` for Codex, `sometimes` for Claude — only
provider versions that emit a per-child `agentId` name the child — and `never`
for the remaining sources.

`getSessionTree` returns pre-order `nodes` (the root first), the `unlinked`
evidence found at any depth, and `diagnostics`. `nodes[0]` is always the root,
so a session with no children — and a database that does not exist yet — comes
back as a one-node tree rather than an empty one. Children are ordered by
`(spawnedAtMs, relationshipUid)` with null spawn times last, so repeated calls
against the same database return identical results.

Traversal visits each session once, at the position pre-order first reaches it.
An edge back into the current branch's ancestry is a cycle and emits a
`RELATIONSHIP_CYCLE` diagnostic instead of looping; an edge to a session already
emitted on another branch is a diamond and is quietly not expanded twice.
`maxDepth` (default 32, maximum 64) and `maxNodes` (default 1000, maximum
10000) bound the work and set `truncated` with a
`RELATIONSHIP_TREE_DEPTH_LIMIT` or `RELATIONSHIP_TREE_TRUNCATED` diagnostic
rather than returning a silently short tree. Tree-level `truncated` means a
budget cut the walk short — a cycle or diamond never sets it — while a node's
own `truncated` marks children it did not expand.

For large topologies, `getSessionChildrenPage` is the keyset primitive and
`sessionDescendants` is the async iterator over it, walking descendants
breadth-first without materializing a tree. It yields the nodes
`getSessionTree` emits minus the root, with the same `childCount` (linked
children only, at the depth boundary included) and the same `truncated` rule.
The order differs: the walker is breadth-first, the tree is pre-order, so a
consumer that depends on either the root node or `getSessionTree`'s ordering
has to account for that.

Native loading failures distinguish unsupported platforms, missing optional
platform packages, addon load failures, SDK/native contract mismatches, and
database open failures through stable `RelayHistoryError` subclasses. Provider
capability failures use `UnsupportedOperationError`.

The old synchronous `AiHist` class and `openAiHist()` API were removed in 1.0.
See [the migration guide](https://github.com/AgentWorkforce/relayhistory/blob/main/docs/native-sdk-migration.md).
