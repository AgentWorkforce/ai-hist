# ai-hist-mcp

Thin `npx` wrapper for the MCP server shipped by the public `ai-hist` SDK:

```bash
npx -y ai-hist-mcp
```

The server imports only `ai-hist` public functions. It never opens SQLite,
loads the native addon directly, scans provider files, or invokes a CLI.

Tools: `search_history`, `recent_history`, `list_sessions`,
`discover_sessions`, `hydrate_session`, `get_session`, `get_session_events`,
`get_session_relationships`, `get_session_tree`, `get_session_tool_calls`,
`get_session_file_edits`, `history_stats`, and
`sync`. Search, recent history, session listing, discovery, statistics, and sync accept a
`scope` of `local`, `remote`, or `all`; scope defaults to `local`.
`get_session`, `get_session_events`, `get_session_relationships`, and
`get_session_tree` address one session by identity and take no `scope`.
`get_session_tool_calls` and `get_session_file_edits` are bounded, cursor-paged
reads that require both a `source` and a `session_id`, because provider session
IDs collide.

Cached reads support all three scopes. Remote acquisition runs through
provider connectors (claude.ai/code web sessions, Codex cloud tasks) that are
configured by the provider CLI's own sign-in on the machine; explicit `remote`
acquisition returns `UNSUPPORTED_OPERATION` when none is configured, while
`all` runs local adapters plus every configured connector. The discovery and
sync tools are therefore annotated as open-world writes.
`hydrate_session` is an idempotent write that fully indexes one previously
discovered identity and optionally its related provider-native sessions from
local provider evidence, so it stays annotated as a local, closed-world write.

`get_session_relationships` and `get_session_tree` read the delegation topology
recorded by hydration and sync: who delegated to whom, what evidence
established the link, whether the child has a stable identity, and whether its
events are independently addressable. Tree traversal is cycle-safe,
deterministically ordered, and bounded by `max_depth` and `max_nodes`.
