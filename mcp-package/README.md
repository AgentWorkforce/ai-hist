# ai-hist-mcp

Thin `npx` wrapper for the MCP server shipped by the public `ai-hist` SDK:

```bash
npx -y ai-hist-mcp
```

The server imports only `ai-hist` public functions. It never opens SQLite,
loads the native addon directly, scans provider files, or invokes a CLI.

Tools: `search_history`, `recent_history`, `list_sessions`,
`discover_sessions`, `get_session`, `get_session_events`, `history_stats`, and
`sync`.
