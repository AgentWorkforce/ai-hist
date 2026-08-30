# Pair hooks (legacy)

The pre-1.0 Pair setup flow has been retired. RelayHistory 1.0 does not expose
`ai-hist pair`, the `pair_check` MCP tool, or an `ai-hist-mcp setup` command.
Existing hook and MCP configurations that call those commands should be
removed; they cannot be repaired by reinstalling the npm packages.

The supported MCP server is a thin adapter over the native-backed public SDK.
It provides local search, recent history, catalog listing and discovery,
session and event reads, statistics, and explicit local sync. See
[Getting started](getting-started.md) for installation and configuration.
