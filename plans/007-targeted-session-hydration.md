# Targeted session hydration

## Shared ingestion

- Keep provider parsing in Rust. Targeted hydration calls the same per-session
  Claude, Codex, Cursor, Grok, and OpenCode ingestion functions used by global
  sync; global walks remain responsible only for enumeration and checkpointing.
- Resolve file-backed sessions from the catalog's local `session_presences`
  locator and verify that the canonical path is under the configured provider
  root before opening it. OpenCode resolves the catalog identity with bounded
  `WHERE session_id = ?` queries against the live read-only database.

## Incrementality and atomicity

- Add a per-session hydration checkpoint containing the indexed provider stamp,
  parser version, last event time, and indexed byte/record diagnostics.
- Read provider evidence before opening the destination write transaction where
  practical. Persist one selected session atomically; existing unique prompt and
  event identities make retries idempotent.
- An unchanged provider stamp returns `unchanged`. A changed live transcript is
  re-read through the shared idempotent ingester and returns `updated`; complete
  JSONL records are consumed while a partial trailing record is ignored.

## Related sessions

- Add an explicit `session_relationships` table keyed by provider-native
  parent-child session IDs. Codex related rollouts are resolved from native root
  session metadata and ingested without flattening child events or promoting
  child task prompts to human prompt history. Claude sidechain evidence remains
  attributed according to the existing parser until the provider exposes a
  stable distinct child identity.

## Concurrency

- Serialize hydration with the existing per-database acquisition lock so it can
  safely overlap catalog reads and cannot race global sync into duplicate work.
  SQLite WAL/busy-timeout behavior continues to serve cache-only readers.

## Public call graph

- Expose typed request/result objects from Rust through N-API. The TypeScript
  SDK validates arguments, applies defaults, checks contract versions, and maps
  stable errors. CLI `sessions hydrate` and MCP `hydrate_session` call only the
  SDK operation.
