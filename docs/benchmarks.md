# Native-path benchmarks

Run benchmarks from the repository root against the default RelayHistory
database:

```bash
npm run benchmark
```

Write the report to a file with:

```bash
npm run benchmark -- --output=foo.md
```

Show a compact terminal table with only the benchmark name and milliseconds:

```bash
npm run benchmark -- --pretty
```

`--pretty` can be combined with `--output` to save the full report while
showing the compact table in the terminal.

The output format follows the extension: `.md` writes a Markdown report and
any other extension writes JSON. Omit `--output` to print JSON to stdout. npm
requires the `--` separator before benchmark-specific options.

Useful overrides:

```bash
npm run benchmark -- --output=foo.md --db=/path/to/ai-history.db
npm run benchmark -- --output=foo.json --discovery-limit=250
AI_HIST_DB=/path/to/ai-history.db npm run benchmark -- --output=foo.md
```

The script measures warm catalog pages of 20 and 100 rows, unchanged and first
shallow discovery, one 200-event page, CLI startup plus catalog listing, and an
MCP `tools/call` catalog request. Results include database file size so a sparse
or real database larger than 2 GiB can demonstrate the key invariant: query
cost follows requested rows/pages, not total file bytes.

For a sparse validation fixture, copy a real migrated database and extend the
file sparsely on a filesystem that supports sparse files. Do not run this test
in ordinary CI; use the opt-in `AI_HIST_LARGE_DB` path and verify catalog and
event queries against it.

## 2026-08-30 baseline

Measured on macOS 26.5 arm64 with Node.js 22.22.2 against a real 3,030,155,264
byte RelayHistory database with an active WAL:

| Operation | Time | Rows/work |
|---|---:|---:|
| warm cache-only catalog | 1.18 ms | 20 rows |
| warm cache-only catalog | 5.67 ms | 100 rows |
| event page | 9.64 ms | 137 rows requested with a 200 cap |
| first shallow discovery | 1,064.44 ms | 100 rows; 57.0 MiB bounded reads |
| unchanged shallow discovery | 34.07 ms | 100 rows; zero files opened |
| CLI startup + catalog | 46.74 ms | 20 rows |
| MCP `list_sessions` | 7.93 ms | 20 rows |

The 20-row catalog query did not load or copy the 2.8 GiB database: Rust opened
it directly and returned the indexed page. Event retrieval likewise returned a
bounded page.
