# ai-hist-mcp

Thin `npx` wrapper for the MCP server shipped by the public `ai-hist` SDK:

```bash
npx -y ai-hist-mcp
```

The server imports only `ai-hist` public functions. It never opens SQLite,
loads the native addon directly, scans provider files, or invokes a CLI.

Tools: `search_history`, `recent_history`, `list_sessions`,
`discover_sessions`, `get_session`, `get_session_events`, `history_stats`, and
`sync`. Search, recent history, session listing, discovery, statistics, and sync accept a
`scope` of `local`, `remote`, or `all`; scope defaults to `local`.

Cached reads support all three scopes. Remote acquisition runs through
provider connectors (claude.ai/code web sessions, Codex cloud tasks) that are
configured by the provider CLI's own sign-in on the machine; explicit `remote`
acquisition returns `UNSUPPORTED_OPERATION` when none is configured, while
`all` runs local adapters plus every configured connector. The discovery and
sync tools are therefore annotated as open-world writes.
