# Cloud sync (legacy)

The standalone cloud-sync workflow documented before RelayHistory 1.0 has
been retired. The npm CLI does not provide `login`, `admin-mint`, `push`, or
service-installation commands, and the old curl installer no longer exists.

Do not use pre-1.0 setup snippets copied from older releases. Local history is
available through the native-backed SDK, CLI, and MCP server documented in
[Getting started](getting-started.md). Run `ai-hist sync` for explicit local
full ingestion or `ai-hist sessions discover` for shallow catalog discovery.

Agent Relay's in-process capture integration is an internal host API, not a
replacement public cloud-sync command. A future public cloud workflow should
be documented here only when it ships as a supported native-backed surface.
