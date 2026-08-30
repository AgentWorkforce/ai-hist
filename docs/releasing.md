# Release and platform validation

All public packages use one version: `ai-hist`, `ai-hist-native`, each native
platform package, and `ai-hist-mcp`. The SDK checks native contract version 2
at initialization.

The `Publish RelayHistory npm packages` workflow:

1. builds Rust and all seven native targets;
2. executes the addon on native CI runners where the target matches;
3. collects `.node` artifacts;
4. creates and publishes per-platform packages;
5. publishes `ai-hist-native` with exact optional dependencies;
6. builds and publishes `ai-hist` (SDK + CLI);
7. publishes `ai-hist-mcp` last;
8. performs a clean registry install and CLI/MCP smoke test.

The SDK root is never published before required platform artifacts. This is
important because npm multi-package publication is not atomic.

## Supported matrix

- Node.js: minimum 20; tested 20 and 22.
- Node-API: level 4.
- macOS: 12+, arm64 and x64.
- Linux: glibc and musl, arm64 and x64.
- Windows: Windows 10/11 and Server 2022, x64 MSVC.
- Windows arm64: unsupported until an executable CI test is reliable.

To validate a local artifact:

```bash
cargo test --workspace
cd crates/ai-hist-napi && npm ci && npm run build:debug
node -e "const n=require('./index.js'); console.log(n.nativeContractVersion())"
cd ../../sdk-ts && npm install && npm test && npm pack
```
