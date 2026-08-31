# Native-path benchmarks

Run benchmarks from the repository root against the default RelayHistory
database:

```bash
npm run benchmark
```

Benchmarks require a release build of the native addon and refuse a debug one
(`npm run build --prefix crates/ai-hist-napi` rebuilds it as release).

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
AI_HIST_DB=/path/to/ai-history.db npm run benchmark -- --output=foo.md
```

## Benchmark definitions

Discovery benchmarks use a generated home directory containing 1,000 valid,
minimal Claude session transcripts and an opencode SQLite store
(`.local/share/opencode/opencode.db`) holding 1,000 minimal sessions with one
message and one part each. Setup, fixture creation, and source-file changes
happen outside the timed regions. The fixture makes each run repeatable and
lets the changed-source case avoid modifying real provider history. The
opencode store is written with `node:sqlite`, so benchmarks need Node 22.13+.

Unprefixed discovery benchmarks read the Claude fixture; the
`opencode`-prefixed ones read the opencode fixture through the same
SDK -> N-API -> Rust path. The number at the end of a benchmark name is the
requested session or event limit, not a cache size.

| Benchmark | Setup outside timing | Timed work |
|---|---|---|
| `cold shallow discovery N` | Fresh fixture; database does not exist | Create/migrate the database, enumerate the 1,000 candidates, shallow-read the newest N files, and upsert N catalog rows through SDK -> N-API -> Rust |
| `unchanged shallow discovery N` | Run one cold discovery of N into a fresh database | Enumerate candidates, match source stamps, and return N cached rows without opening unchanged transcripts |
| `cold->changed shallow discovery N` | Run one cold discovery of N, then append a valid assistant record to those N fixture transcripts | Enumerate candidates, detect changed stamps, shallow-read the N changed files, and update their catalog rows |
| `opencode cold shallow discovery N` | Fresh fixture; database does not exist | Create/migrate the database, snapshot the opencode store, enumerate the 1,000 sessions, shallow-read the newest N from the snapshot, and upsert N catalog rows |
| `opencode unchanged shallow discovery N` | Run one cold discovery of N into a fresh database | Snapshot the store, enumerate sessions, match `time_created`/`time_updated` stamps, and return N cached rows |
| `opencode cold->changed shallow discovery N` | Run one cold discovery of N, then insert an assistant message and bump `time_updated` for those N sessions | Snapshot the store, detect the N changed stamps, re-read those sessions, and update their catalog rows |
| `warm session events N` | Select a full session from the configured real database and make one untimed 200-event request | Open the database and return up to N cached events through SDK -> N-API -> Rust; provider transcripts are not read |
| `CLI startup + cold shallow discovery 20` | Fresh database and the generated fixture | Start a new Node process, load the CLI and native addon, then perform cold discovery of 20 sessions and serialize JSON |
| `MCP cold shallow discovery 20` | Start and initialize the MCP server with a fresh database and the generated fixture | Perform one MCP `tools/call` round trip that cold-discovers 20 sessions; MCP process startup and initialization are excluded |

The report includes the real database file size used by the event benchmarks,
so a database larger than 2 GiB can demonstrate that event-query cost follows
the requested page rather than total database bytes.

For a sparse validation fixture, copy a real migrated database and extend the
file sparsely on a filesystem that supports sparse files. Do not run this test
in ordinary CI; use the opt-in `AI_HIST_LARGE_DB` path and verify catalog and
event queries against it.

## 2026-08-31 baseline

Measured on macOS arm64 (Apple M2 Max) with Node.js 22.22.2 against a real
3,131,932,672 byte RelayHistory database with an active WAL:

| Operation | Time | Rows/work |
|---|---:|---:|
| cold shallow discovery | 13.94 ms | 20 rows |
| cold shallow discovery | 18.26 ms | 100 rows |
| cold shallow discovery | 79.66 ms | 1,000 rows |
| unchanged shallow discovery | 4.36 ms | 20 rows; zero files opened |
| unchanged shallow discovery | 5.11 ms | 100 rows; zero files opened |
| unchanged shallow discovery | 17.26 ms | 1,000 rows; zero files opened |
| cold->changed shallow discovery | 6.29 ms | 20 changed rows |
| cold->changed shallow discovery | 12.58 ms | 100 changed rows |
| cold->changed shallow discovery | 77.61 ms | 1,000 changed rows |
| opencode cold shallow discovery | 13.61 ms | 20 rows |
| opencode cold shallow discovery | 36.77 ms | 100 rows |
| opencode cold shallow discovery | 284.92 ms | 1,000 rows |
| opencode unchanged shallow discovery | 2.15 ms | 20 rows; zero shallow reads |
| opencode unchanged shallow discovery | 6.61 ms | 100 rows; zero shallow reads |
| opencode unchanged shallow discovery | 12.44 ms | 1,000 rows; zero shallow reads |
| opencode cold->changed shallow discovery | 7.60 ms | 20 changed rows |
| opencode cold->changed shallow discovery | 32.84 ms | 100 changed rows |
| opencode cold->changed shallow discovery | 372.95 ms | 1,000 changed rows |
| warm session events | 0.66 ms | 20 rows |
| warm session events | 1.32 ms | 200 rows |
| CLI startup + cold shallow discovery | 48.62 ms | 20 rows |
| MCP cold shallow discovery | 20.02 ms | 20 rows |

The event queries did not load or copy the 2.9 GiB database: Rust opened it
directly and returned a bounded indexed page. Every opencode run, including
unchanged, snapshots the whole store once with SQLite backup, so its
`filesOpened` is always 1 and its `bytesRead` is the store's size.
