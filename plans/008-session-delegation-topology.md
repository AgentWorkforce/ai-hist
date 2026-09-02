# Session delegation topology

## Relationship model

- Record one row per observed delegation in `session_relationships`, keyed by
  `(source, parent_session_id, relationship_uid)`. The uid is
  `child:<child_session_id>` when the child is named and
  `evidence:<evidence_kind>:<evidence_locator>` when it is not, so repeated
  ingestion is idempotent and each unlinked sidecar keeps its own row.
- Keep `child_session_id` nullable and pair it with `identity_status`
  (`observed` or `unlinked`). A child identity is never synthesized, inferred,
  or taken from a file name.
- Store what the provider actually recorded beside the edge: child agent type,
  name, model, spawn depth, spawn time, and whether the child's events are
  independently addressable.

## Provider evidence

- Carry the evidence that established every link: `evidence_kind`, the provider
  file in `evidence_locator`, and the provider-native reference in
  `evidence_ref` — a Claude `toolUseId`, a Codex `parent_thread_id`.
- Codex names every child thread in its rollout metadata, so its children are
  always observed; both targeted hydration and global sync record them.
- Claude subagent transcripts carry the parent's `sessionId` plus a per-child
  `agentId` on newer provider versions. With that id the child's events are
  indexed under it; without it the delegation is recorded as unlinked evidence
  and the sidechain output stays attributed to the parent.
- A delegated instruction is not a human prompt and a delegated thread is not a
  top-level session: neither becomes a `history` row or a catalog session.

## Traversal

- Order children by `(spawned_at_ms, relationship_uid)` with null spawn times
  last. The uid is unique per parent, so that is a total order shared by the
  paged children query and the tree walk.
- Walk pre-order over an explicit stack. A session is emitted once, at its
  shallowest reachable depth. A repeat is a cycle only when the edge points
  back into the current branch's own ancestry: that marks its parent truncated
  and emits a cycle diagnostic instead of looping. A repeat that reaches a
  session already emitted on another branch is a diamond — nothing is missing,
  so it is simply not expanded again, and nothing is diagnosed, truncated, or
  charged against the node budget.
- Bound the work with `max_depth` and `max_nodes`, one indexed child query per
  emitted node, and report depth-limit, truncation, and unlinked-child
  diagnostics rather than returning a silently short tree.

## Public call graph

- Expose `session_relationships`, `session_tree`, and `session_children_page`
  from Rust through N-API under their own contract version. The TypeScript SDK
  validates identities, normalizes nullable fields, checks the contract, and
  adds the `sessionDescendants` and `sessionEventsIncludingDescendants`
  iterators over the paged primitive. Every yielded event keeps its true owning
  session id.
- CLI `sessions relationships` / `sessions tree` and MCP
  `get_session_relationships` / `get_session_tree` call only the SDK
  operations, and address a session by identity rather than by scope.
