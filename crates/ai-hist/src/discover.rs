//! Fast, shallow coding-agent session discovery.
//!
//! Two operations live here, and they are deliberately distinct:
//!
//! * [`list_session_catalog`] — a cache-only query over the `sessions` table.
//!   It never touches a provider transcript and never reads `history`,
//!   `session_events` or `tool_calls`. It is what a desktop app calls on every
//!   paint.
//! * [`discover_sessions`] — inspects the known provider locations, extracts
//!   minimal metadata with *bounded* reads, upserts the `sessions` catalog and
//!   emits rows progressively in global recency order.
//!
//! # Fidelity model
//!
//! Every value in [`ShallowSession`] is one of three things, and the
//! distinction is part of the contract:
//!
//! * **Observed** — read straight out of provider data (`session_id`, `cwd`,
//!   `git_branch`, `originator`, `agent_version`, `repo_url`,
//!   `initial_commit`, `workspace_roots`, `models`, and any timestamp the
//!   provider actually records).
//! * **Derived** — computed deterministically by RelayHistory from provider
//!   data. `first_prompt` is the only derived field: it is a bounded excerpt
//!   (see [`EXCERPT_MAX_CHARS`]) of the first *substantive* human turn, with
//!   provider control/meta turns skipped. `last_activity_ms` is derived from
//!   the filesystem mtime for providers that record no timestamps (cursor).
//! * **Unavailable until full indexing** — anything not in this struct. Tool
//!   calls, file edits, per-message events, token spend, and the full
//!   transcript body all require `ai-hist sync`. [`ShallowSession::discovery_state`]
//!   says which of the two a row is: `"shallow"` or `"full"`.
//!
//! Absent metadata is `None`. Nothing here is ever invented to fill a column.
//!
//! # Product boundary
//!
//! Discovery reports *which sessions exist* and identifying metadata. It does
//! not infer project membership, work status, health, risk, or success, and it
//! does not summarize outcomes.
//!
//! # Concurrency
//!
//! Discovery deliberately does **not** take the `sync` advisory lock. Every
//! write it performs is an idempotent, stamp-guarded upsert into `sessions`,
//! so a concurrent `ai-hist sync` and a concurrent `discover` converge: the
//! full-sync path only ever upgrades a row to `discovery_state = 'full'`, and
//! the shallow path never downgrades one. Writes go through the normal
//! busy-retry connection.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

use ai_hist_core::SOURCE_CHOICES;
use anyhow::{Context, Result};
use rusqlite::{params, Connection, DatabaseName, OpenFlags};
use serde::Serialize;
use serde_json::Value;

/// Version of the machine-readable session-catalog contract.
///
/// Bumped when the shape or meaning of [`ShallowSession`] / the CLI JSON
/// payloads changes in a way a consumer must notice.
pub const SESSION_CATALOG_CONTRACT_VERSION: u32 = 1;

/// Version of the shallow scanners themselves.
///
/// Persisted as the `v{N}:` prefix of `sessions.source_stamp`. Bumping it
/// invalidates every stored stamp, so a scanner that learns to extract a new
/// field re-reads sources whose bytes never changed. `parser_version` keeps its
/// existing meaning (full-ingest parser generation) and is untouched.
pub const SHALLOW_SCANNER_VERSION: u32 = 1;

/// Most bytes a shallow head read may consume from one transcript.
pub const HEAD_SCAN_MAX_BYTES: u64 = 256 * 1024;
/// Most complete JSONL records a shallow head read may consider.
pub const HEAD_SCAN_MAX_LINES: usize = 400;
/// Most bytes a shallow tail read may consume from one transcript.
pub const TAIL_SCAN_MAX_BYTES: u64 = 64 * 1024;
/// Character cap for stored text excerpts, matching `last_assistant_text`.
pub const EXCERPT_MAX_CHARS: usize = 4096;
/// Default row cap for `list_session_catalog` when the caller gives none.
pub const DEFAULT_CATALOG_LIMIT: i64 = 50;

/// Catalog row: one coding-agent session as shallow discovery knows it.
///
/// See the module docs for which fields are observed and which are derived.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ShallowSession {
    /// Provider that owns the session (`claude`, `codex`, …). Observed.
    pub source: String,
    /// The provider's own session identifier. Observed. Stable across
    /// rescans; `(source, session_id)` is the catalog primary key, so the same
    /// native id under two providers is two distinct rows.
    pub session_id: String,
    /// Working directory the provider reported for the session. Observed.
    /// Per provider: claude — `cwd` from the first transcript record; codex —
    /// `session_meta.payload.cwd`; grok — `summary.json` `info.cwd`/`git_root_dir`,
    /// else the percent-decoded project directory; cursor — decoded from the
    /// project directory name; opencode — `session.directory`; relay — none
    /// (a relay thread has no working directory).
    pub cwd: Option<String>,
    /// Git branch the provider reported, last observed value. Observed.
    pub git_branch: Option<String>,
    /// Earliest activity timestamp the provider records. Observed. `None`
    /// when the provider records no timestamps at all (cursor).
    pub first_activity_ms: Option<i64>,
    /// Latest activity timestamp. Observed where the provider records one;
    /// filesystem-derived (file mtime) for cursor, which records none.
    pub last_activity_ms: Option<i64>,
    /// Bounded excerpt of the first substantive human prompt. **Derived.**
    pub first_prompt: Option<String>,
    /// Bounded excerpt of the last assistant text. Observed; only populated by
    /// the full-ingest path, so a purely shallow row leaves it `None`.
    pub last_assistant_text: Option<String>,
    /// Model ids observed in the bounded read. Observed, best effort: never a
    /// reason to widen a read, so an empty list means "not seen cheaply", not
    /// "no model".
    pub models: Vec<String>,
    /// Client that originated the session (codex `session_meta.originator`).
    /// Observed.
    pub originator: Option<String>,
    /// Agent CLI version (codex `cli_version`, claude record `version`).
    /// Observed.
    pub agent_version: Option<String>,
    /// Repository remote URL, when the provider records one. Observed.
    pub repo_url: Option<String>,
    /// Commit the session started from, when the provider records one. Observed.
    pub initial_commit: Option<String>,
    /// Extra workspace roots, when the provider records them. Observed.
    pub workspace_roots: Vec<String>,
    /// Path of the provider file this row came from, when it is file-backed.
    pub raw_path: Option<String>,
    /// Change stamp of the raw source at scan time, `v{scanner}:{provider stamp}`.
    pub source_stamp: Option<String>,
    /// `"shallow"` (catalog row only) or `"full"` (full evidence ingested).
    pub discovery_state: String,
    /// `true` when this row was served from the catalog without re-reading the
    /// provider source — either a cache-only list, or a rescan whose stamp
    /// matched.
    pub from_cache: bool,
}

/// A cheaply enumerated discovery candidate.
///
/// Producing one must not read file contents: a directory walk plus `stat`,
/// or one small indexed query for the database-backed providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Provider that produced the candidate.
    pub source: &'static str,
    /// Opaque provider-scoped handle: a file path, or a session id for the
    /// database-backed providers.
    pub locator: String,
    /// Session id when enumeration already knows it without reading anything
    /// (cursor, opencode, relay); `None` when only the shallow read can say.
    pub session_id: Option<String>,
    /// Recency signal used for global ordering — a provider timestamp where
    /// one is available for free, else the file mtime.
    pub recency_hint_ms: Option<i64>,
    /// Raw provider change marker; stored as `v{scanner}:{stamp}`.
    pub stamp: String,
}

/// Counters describing the work one discovery run actually did.
///
/// Exposed so callers (and tests) can assert bounded behaviour without a wall
/// clock: a limited request must not read the whole archive, a cache-only list
/// must open zero files, and an unchanged rescan must perform zero shallow
/// reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct DiscoveryCounters {
    /// Candidates produced by enumeration, before the global limit.
    pub candidates_enumerated: u64,
    /// Candidates whose source was actually read.
    pub shallow_reads: u64,
    /// Candidates served from the catalog because their stamp was unchanged.
    pub skipped_unchanged: u64,
    /// Provider files (and database snapshots) opened for reading.
    pub files_opened: u64,
    /// Bytes read out of provider sources.
    pub bytes_read: u64,
}

/// Environment one discovery run operates in: where the provider data lives,
/// the catalog connection, and the run's counters.
pub struct DiscoveryEnv<'a> {
    /// Home directory the file-backed providers are rooted at.
    pub home: PathBuf,
    /// Path to the opencode database.
    pub opencode_db: PathBuf,
    conn: &'a Connection,
    counters: Cell<DiscoveryCounters>,
}

impl<'a> DiscoveryEnv<'a> {
    /// Build an environment from the process environment (`HOME`, `OPENCODE_DB`).
    pub fn new(conn: &'a Connection) -> Self {
        Self {
            home: crate::home_dir(),
            opencode_db: crate::default_opencode_db_path(),
            conn,
            counters: Cell::new(DiscoveryCounters::default()),
        }
    }

    /// Build an environment with explicit roots, for hosts that keep provider
    /// data somewhere other than `$HOME` (and for tests, which must not mutate
    /// process-wide environment variables).
    pub fn with_roots(conn: &'a Connection, home: PathBuf, opencode_db: PathBuf) -> Self {
        Self {
            home,
            opencode_db,
            conn,
            counters: Cell::new(DiscoveryCounters::default()),
        }
    }

    /// The catalog connection. `relay` discovers from already-synced local
    /// rows through this and never opens a socket.
    pub fn conn(&self) -> &Connection {
        self.conn
    }

    /// Counters accumulated so far.
    pub fn counters(&self) -> DiscoveryCounters {
        self.counters.get()
    }

    fn note_open(&self) {
        let mut counters = self.counters.get();
        counters.files_opened += 1;
        self.counters.set(counters);
    }

    fn note_bytes(&self, bytes: u64) {
        let mut counters = self.counters.get();
        counters.bytes_read += bytes;
        self.counters.set(counters);
    }

    fn note_candidates(&self, count: u64) {
        let mut counters = self.counters.get();
        counters.candidates_enumerated += count;
        self.counters.set(counters);
    }

    fn note_shallow_read(&self) {
        let mut counters = self.counters.get();
        counters.shallow_reads += 1;
        self.counters.set(counters);
    }

    fn note_skipped(&self) {
        let mut counters = self.counters.get();
        counters.skipped_unchanged += 1;
        self.counters.set(counters);
    }
}

/// One provider's shallow adapter.
///
/// Implementations must be cheap: [`enumerate`](ShallowSessionProvider::enumerate)
/// may stat but not read, and [`read_shallow`](ShallowSessionProvider::read_shallow)
/// must stay inside [`HEAD_SCAN_MAX_BYTES`] / [`TAIL_SCAN_MAX_BYTES`] per
/// source. Returning `Ok(None)` from `read_shallow` means "this candidate is
/// not a session" (a codex subagent thread, a file with no usable metadata) —
/// it is not an error.
pub trait ShallowSessionProvider {
    /// The `SOURCE_CHOICES` name this adapter covers.
    fn source(&self) -> &'static str;
    /// Cheap enumeration: directory walk + stat, or one indexed query.
    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>>;
    /// Bounded read of one candidate into a catalog row.
    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>>;
}

/// A `SOURCE_CHOICES` entry that deliberately has no shallow adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SourceExemption {
    /// The exempt source name.
    pub source: &'static str,
    /// Why it has no adapter. Shown in machine-readable summaries.
    pub reason: &'static str,
}

/// Sources with no shallow adapter, and why.
///
/// Paired with [`shallow_providers`] by a registry regression test that
/// asserts every `SOURCE_CHOICES` entry is covered by exactly one of the two
/// lists — so adding a provider to `SOURCE_CHOICES` fails the build until
/// someone decides whether it is discoverable.
pub const DISCOVERY_EXEMPTIONS: &[SourceExemption] = &[SourceExemption {
    source: "trajectory",
    reason: "derived trajectory records, not provider sessions",
}];

/// Every shallow adapter, one per discoverable source.
pub fn shallow_providers() -> Vec<Box<dyn ShallowSessionProvider>> {
    vec![
        Box::new(ClaudeProvider),
        Box::new(CodexProvider),
        Box::new(CursorProvider),
        Box::new(GrokProvider),
        Box::new(OpencodeProvider::default()),
        Box::new(RelayProvider),
    ]
}

// ---------------------------------------------------------------------------
// bounded reads
// ---------------------------------------------------------------------------

/// The complete JSONL records a bounded read recovered from one file.
///
/// Only newline-terminated records are returned, matching the project's
/// incomplete-record convention: a transcript being written right now has a
/// partial trailing line, and that line is not yet a record.
struct BoundedJsonl {
    head: Vec<String>,
    tail: Vec<String>,
}

fn complete_lines(buffer: &[u8], drop_leading_fragment: bool) -> Vec<String> {
    let text = String::from_utf8_lossy(buffer);
    let mut lines: Vec<String> = text.split_inclusive('\n').map(str::to_string).collect();
    if lines.last().is_some_and(|line| !line.ends_with('\n')) {
        lines.pop();
    }
    if drop_leading_fragment && !lines.is_empty() {
        lines.remove(0);
    }
    lines
        .into_iter()
        .map(|line| line.trim_end_matches(['\n', '\r']).to_string())
        .filter(|line| !line.trim().is_empty())
        .collect()
}

/// Read the head (and, for a large file, the tail) of a JSONL transcript
/// without ever reading the whole thing.
///
/// One file handle regardless of size. Files inside the head budget are read
/// once and serve as their own tail.
fn read_bounded_jsonl(env: &DiscoveryEnv<'_>, path: &Path) -> Result<BoundedJsonl> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    env.note_open();
    let len = file.metadata()?.len();
    if len <= HEAD_SCAN_MAX_BYTES {
        let mut buffer = Vec::with_capacity(len as usize);
        file.read_to_end(&mut buffer)?;
        env.note_bytes(buffer.len() as u64);
        let mut lines = complete_lines(&buffer, false);
        let tail = lines.clone();
        lines.truncate(HEAD_SCAN_MAX_LINES);
        return Ok(BoundedJsonl { head: lines, tail });
    }
    let mut head_buffer = vec![0u8; HEAD_SCAN_MAX_BYTES as usize];
    let mut filled = 0usize;
    while filled < head_buffer.len() {
        let read = file.read(&mut head_buffer[filled..])?;
        if read == 0 {
            break;
        }
        filled += read;
    }
    head_buffer.truncate(filled);
    env.note_bytes(filled as u64);
    let mut head = complete_lines(&head_buffer, false);
    head.truncate(HEAD_SCAN_MAX_LINES);

    let tail_start = len.saturating_sub(TAIL_SCAN_MAX_BYTES);
    file.seek(SeekFrom::Start(tail_start))?;
    let mut tail_buffer = Vec::with_capacity(TAIL_SCAN_MAX_BYTES as usize);
    file.take(TAIL_SCAN_MAX_BYTES)
        .read_to_end(&mut tail_buffer)?;
    env.note_bytes(tail_buffer.len() as u64);
    let tail = complete_lines(&tail_buffer, tail_start > 0);
    Ok(BoundedJsonl { head, tail })
}

fn excerpt(text: &str) -> String {
    text.trim().chars().take(EXCERPT_MAX_CHARS).collect()
}

fn json_lines(lines: &[String]) -> impl Iterator<Item = Value> + '_ {
    lines
        .iter()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
}

/// A JSON array column value, or `None` when there is nothing observed to
/// store. An empty list is never written as `[]` — absent stays absent.
fn json_array_or_none(values: &[String]) -> Option<String> {
    (!values.is_empty()).then(|| serde_json::to_string(values).unwrap_or_else(|_| "[]".into()))
}

fn push_unique(models: &mut Vec<String>, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|s| !s.is_empty()) {
        if !models.iter().any(|existing| existing == value) {
            models.push(value.to_string());
        }
    }
}

fn text_of(content: Option<&Value>) -> Option<String> {
    let content = content?;
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }
    let items = content.as_array()?;
    let parts: Vec<&str> = items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

// ---------------------------------------------------------------------------
// claude
// ---------------------------------------------------------------------------

struct ClaudeProvider;

/// Wrappers Claude Code writes into the user role that are not human turns.
const CLAUDE_CONTROL_PREFIXES: &[&str] = &[
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<user-prompt-submit-hook>",
    "Caveat: The messages below were generated by the user while running local commands",
];

fn claude_substantive_prompt(value: &Value) -> Option<String> {
    let role = value
        .pointer("/message/role")
        .and_then(Value::as_str)
        .or_else(|| value.get("type").and_then(Value::as_str));
    if role != Some("user") {
        return None;
    }
    // Sidechain rows are the parent agent's own prompts to a subagent, and
    // meta rows are Claude Code's bookkeeping. Neither is a human turn.
    if value.get("isSidechain").and_then(Value::as_bool) == Some(true)
        || value.get("isMeta").and_then(Value::as_bool) == Some(true)
    {
        return None;
    }
    let text = text_of(value.pointer("/message/content"))?;
    let trimmed = text.trim();
    if trimmed.is_empty()
        || CLAUDE_CONTROL_PREFIXES
            .iter()
            .any(|prefix| trimmed.starts_with(prefix))
    {
        return None;
    }
    Some(excerpt(trimmed))
}

fn claude_timestamp(value: &Value) -> Option<i64> {
    value.get("timestamp").and_then(|v| {
        v.as_str()
            .and_then(crate::parse_iso_ms)
            .or_else(|| v.as_i64())
    })
}

impl ShallowSessionProvider for ClaudeProvider {
    fn source(&self) -> &'static str {
        "claude"
    }

    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        file_candidates(
            "claude",
            crate::collect_matching_files(&env.home.join(".claude/projects"), "", "jsonl")?,
            crate::file_stamp,
        )
    }

    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        let path = PathBuf::from(&candidate.locator);
        let bounded = read_bounded_jsonl(env, &path)?;
        let mut session = ShallowSession {
            source: "claude".into(),
            raw_path: Some(candidate.locator.clone()),
            ..Default::default()
        };
        let mut models = Vec::new();
        let mut session_id = None;
        // A subagent sidecar transcript is its own file whose records carry the
        // *parent's* sessionId (see `ingest_claude_transcript`). Enumerating it
        // as a session would emit the parent twice per run and let the two
        // files fight over one row's raw_path/source_stamp, so the stamp never
        // matched again and one of them was re-read forever.
        let mut identified_records = 0usize;
        let mut sidechain_records = 0usize;
        let mut parsed_records = 0usize;
        for value in json_lines(&bounded.head) {
            parsed_records += 1;
            if value
                .get("sessionId")
                .and_then(Value::as_str)
                .is_some_and(|id| !id.is_empty())
            {
                identified_records += 1;
                if value.get("isSidechain").and_then(Value::as_bool) == Some(true) {
                    sidechain_records += 1;
                }
            }
            if session_id.is_none() {
                session_id = value
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
            }
            if session.cwd.is_none() {
                session.cwd = value.get("cwd").and_then(Value::as_str).map(str::to_string);
            }
            if let Some(branch) = value.get("gitBranch").and_then(Value::as_str) {
                session.git_branch = Some(branch.to_string());
            }
            if let Some(version) = value.get("version").and_then(Value::as_str) {
                session.agent_version = Some(version.to_string());
            }
            push_unique(
                &mut models,
                value.pointer("/message/model").and_then(Value::as_str),
            );
            if let Some(ts) = claude_timestamp(&value) {
                session.first_activity_ms.get_or_insert(ts);
                session.last_activity_ms = Some(ts);
            }
            if session.first_prompt.is_none() {
                session.first_prompt = claude_substantive_prompt(&value);
            }
        }
        for value in json_lines(&bounded.tail) {
            if let Some(ts) = claude_timestamp(&value) {
                session.last_activity_ms = Some(ts);
            }
            if let Some(branch) = value.get("gitBranch").and_then(Value::as_str) {
                session.git_branch = Some(branch.to_string());
            }
        }
        // Every identified record in the head belongs to a sidechain: this is a
        // sidecar for a session whose own transcript is enumerated separately.
        // A session's primary transcript always opens with non-sidechain turns,
        // because a subagent can only be spawned by one.
        if identified_records > 0 && sidechain_records == identified_records {
            return Ok(None);
        }
        // A file with complete records that parse as nothing is corrupt, not a
        // session. Publishing it under its file stem would put a fabricated
        // row in the catalog and hide the corruption; a diagnostic names it.
        // A file with no complete records at all is merely empty (a session
        // that has just started) and is simply not a session yet.
        if parsed_records == 0 {
            anyhow::ensure!(
                bounded.head.is_empty(),
                "no parseable JSON records in the first {} record(s)",
                bounded.head.len()
            );
            return Ok(None);
        }
        let Some(session_id) = session_id.or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
                .filter(|s| !s.is_empty())
        }) else {
            return Ok(None);
        };
        session.session_id = session_id;
        session.models = models;
        Ok(Some(session))
    }
}

// ---------------------------------------------------------------------------
// codex
// ---------------------------------------------------------------------------

struct CodexProvider;

impl ShallowSessionProvider for CodexProvider {
    fn source(&self) -> &'static str {
        "codex"
    }

    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        let mut files = Vec::new();
        for root in [
            env.home.join(".codex/sessions"),
            env.home.join(".codex/archived_sessions"),
        ] {
            files.extend(crate::collect_matching_files(&root, "rollout-", "jsonl")?);
        }
        file_candidates("codex", files, crate::file_stamp)
    }

    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        let path = PathBuf::from(&candidate.locator);
        let bounded = read_bounded_jsonl(env, &path)?;
        let Some(meta) = bounded.head.first().and_then(|line| {
            serde_json::from_str::<Value>(line)
                .ok()
                .filter(|v| v.get("type").and_then(Value::as_str) == Some("session_meta"))
        }) else {
            return Ok(None);
        };
        let payload = meta.get("payload");
        let Some(session_id) = payload
            .and_then(|p| p.get("id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            return Ok(None);
        };
        // Subagent threads are real rollouts but not user sessions; the full
        // sync excludes them from `sessions` and so does discovery.
        let is_subagent = payload
            .and_then(|p| p.get("thread_source"))
            .and_then(Value::as_str)
            == Some("subagent")
            || payload
                .and_then(|p| p.get("source"))
                .and_then(Value::as_object)
                .is_some_and(|s| s.contains_key("subagent"));
        if is_subagent {
            return Ok(None);
        }
        let git = payload.and_then(|p| p.get("git"));
        let string_field = |owner: Option<&Value>, key: &str| {
            owner
                .and_then(|o| o.get(key))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let mut models = Vec::new();
        push_unique(
            &mut models,
            payload.and_then(|p| p.get("model")).and_then(Value::as_str),
        );
        let mut first_prompt = None;
        let mut first_activity_ms = claude_timestamp(&meta);
        let mut last_activity_ms = first_activity_ms;
        for value in json_lines(&bounded.head) {
            if let Some(ts) = claude_timestamp(&value) {
                first_activity_ms.get_or_insert(ts);
                last_activity_ms = Some(ts);
            }
            if value.get("type").and_then(Value::as_str) == Some("turn_context") {
                push_unique(
                    &mut models,
                    value.pointer("/payload/model").and_then(Value::as_str),
                );
            }
            if first_prompt.is_none() {
                first_prompt = codex_substantive_prompt(&value);
            }
        }
        for value in json_lines(&bounded.tail) {
            if let Some(ts) = claude_timestamp(&value) {
                last_activity_ms = Some(ts);
            }
        }
        let mtime = crate::file_modified_ms(&path);
        Ok(Some(ShallowSession {
            source: "codex".into(),
            session_id: session_id.to_string(),
            cwd: string_field(payload, "cwd"),
            git_branch: string_field(git, "branch"),
            first_activity_ms,
            last_activity_ms: last_activity_ms.or(mtime),
            first_prompt,
            models,
            originator: string_field(payload, "originator"),
            agent_version: string_field(payload, "cli_version"),
            repo_url: string_field(git, "repository_url")
                .or_else(|| string_field(git, "remote_url")),
            initial_commit: string_field(git, "commit_hash"),
            workspace_roots: string_list(payload.and_then(|p| p.get("workspace_roots"))),
            raw_path: Some(candidate.locator.clone()),
            ..Default::default()
        }))
    }
}

fn codex_substantive_prompt(value: &Value) -> Option<String> {
    if value.get("type").and_then(Value::as_str) != Some("event_msg") {
        return None;
    }
    let payload = value.get("payload")?;
    if payload.get("type").and_then(Value::as_str) != Some("user_message") {
        return None;
    }
    let message = payload.get("message").and_then(Value::as_str)?;
    let trimmed = message.trim();
    if trimmed.is_empty() || crate::is_codex_control_context(trimmed) {
        return None;
    }
    Some(excerpt(trimmed))
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// cursor
// ---------------------------------------------------------------------------

struct CursorProvider;

impl ShallowSessionProvider for CursorProvider {
    fn source(&self) -> &'static str {
        "cursor"
    }

    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        let root = env.home.join(".cursor/projects");
        let mut out = Vec::new();
        for project_dir in crate::sorted_dirs(&root)? {
            let transcripts = project_dir.join("agent-transcripts");
            if !transcripts.is_dir() {
                continue;
            }
            for session_dir in crate::sorted_dirs(&transcripts)? {
                let Some(session_id) = session_dir.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                let jsonl = session_dir.join(format!("{session_id}.jsonl"));
                if !jsonl.is_file() {
                    continue;
                }
                let Ok(stamp) = crate::file_stamp(&jsonl) else {
                    continue;
                };
                out.push(Candidate {
                    source: "cursor",
                    locator: jsonl.to_string_lossy().into_owned(),
                    session_id: Some(session_id.to_string()),
                    recency_hint_ms: crate::file_modified_ms(&jsonl),
                    stamp,
                });
            }
        }
        Ok(out)
    }

    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        let path = PathBuf::from(&candidate.locator);
        let Some(session_id) = candidate.session_id.clone() else {
            return Ok(None);
        };
        let bounded = read_bounded_jsonl(env, &path)?;
        let first_prompt = bounded
            .head
            .iter()
            .filter_map(|line| ai_hist_core::parse_cursor_text(line).ok().flatten())
            .map(|prompt| excerpt(&prompt))
            .find(|prompt| !prompt.is_empty());
        let cwd = path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .map(crate::decode_cursor_project);
        Ok(Some(ShallowSession {
            source: "cursor".into(),
            session_id,
            cwd,
            // Cursor transcripts carry no per-message timestamps at all. The
            // file mtime is the only time signal, so it is reported as
            // last_activity (filesystem-derived) and first_activity stays
            // NULL rather than being invented.
            first_activity_ms: None,
            last_activity_ms: crate::file_modified_ms(&path),
            first_prompt,
            raw_path: Some(candidate.locator.clone()),
            ..Default::default()
        }))
    }
}

// ---------------------------------------------------------------------------
// grok
// ---------------------------------------------------------------------------

struct GrokProvider;

impl ShallowSessionProvider for GrokProvider {
    fn source(&self) -> &'static str {
        "grok"
    }

    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        file_candidates(
            "grok",
            crate::collect_matching_files(
                &env.home.join(".grok/sessions"),
                "chat_history",
                "jsonl",
            )?,
            crate::grok_session_stamp,
        )
    }

    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        let chat = PathBuf::from(&candidate.locator);
        let summary_path = chat.with_file_name("summary.json");
        let summary = if summary_path.is_file() {
            env.note_open();
            env.note_bytes(fs::metadata(&summary_path).map(|m| m.len()).unwrap_or(0));
            crate::read_grok_summary(&summary_path)
        } else {
            None
        };
        let fallback_session = chat
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let session_id = summary
            .as_ref()
            .and_then(|s| s.pointer("/info/id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or(fallback_session);
        if session_id.is_empty() {
            return Ok(None);
        }
        let cwd = summary
            .as_ref()
            .and_then(|s| s.pointer("/info/cwd").or_else(|| s.get("git_root_dir")))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| crate::grok_project_from_path(&chat));
        let git_branch = summary
            .as_ref()
            .and_then(|s| s.get("head_branch"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let first_activity_ms = summary
            .as_ref()
            .and_then(|s| s.get("created_at"))
            .and_then(Value::as_str)
            .and_then(crate::parse_iso_ms);
        let last_activity_ms = summary
            .as_ref()
            .and_then(|s| s.get("updated_at"))
            .and_then(Value::as_str)
            .and_then(crate::parse_iso_ms)
            .or_else(|| crate::file_modified_ms(&chat));
        let mut models = Vec::new();
        push_unique(
            &mut models,
            summary
                .as_ref()
                .and_then(|s| s.pointer("/info/model").or_else(|| s.get("model")))
                .and_then(Value::as_str),
        );
        let mut first_prompt = None;
        if chat.is_file() {
            let bounded = read_bounded_jsonl(env, &chat)?;
            for value in json_lines(&bounded.head) {
                if let Some(text) = crate::grok_chat_text(&value, "user") {
                    first_prompt = Some(excerpt(&text));
                    break;
                }
            }
        }
        Ok(Some(ShallowSession {
            source: "grok".into(),
            session_id,
            cwd,
            git_branch,
            first_activity_ms,
            last_activity_ms,
            first_prompt,
            models,
            raw_path: Some(candidate.locator.clone()),
            ..Default::default()
        }))
    }
}

// ---------------------------------------------------------------------------
// opencode
// ---------------------------------------------------------------------------

/// Shallow adapter over the opencode SQLite store.
///
/// Reads the `session` table directly instead of the full sync's
/// session/message/part join. The database is snapshotted once per run with
/// `Connection::backup`, exactly as the full sync does, so an in-flight WAL
/// never yields a torn read.
#[derive(Default)]
struct OpencodeProvider {
    snapshot: RefCell<Option<tempfile::TempPath>>,
}

impl OpencodeProvider {
    fn ensure_snapshot(&self, env: &DiscoveryEnv<'_>) -> Result<bool> {
        if self.snapshot.borrow().is_some() {
            return Ok(true);
        }
        if !env.opencode_db.exists() {
            return Ok(false);
        }
        let tmp = tempfile::NamedTempFile::new()?.into_temp_path();
        let live = Connection::open_with_flags(
            &env.opencode_db,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )
        .with_context(|| format!("opening {}", env.opencode_db.display()))?;
        live.busy_timeout(Duration::from_secs(5))?;
        live.backup(DatabaseName::Main, &tmp, None)
            .with_context(|| format!("snapshotting {}", env.opencode_db.display()))?;
        env.note_open();
        env.note_bytes(fs::metadata(&env.opencode_db).map(|m| m.len()).unwrap_or(0));
        *self.snapshot.borrow_mut() = Some(tmp);
        Ok(true)
    }

    fn open_snapshot(&self) -> Result<Connection> {
        let guard = self.snapshot.borrow();
        let path = guard
            .as_ref()
            .context("opencode snapshot was not prepared")?;
        Ok(Connection::open(path)?)
    }
}

fn table_columns(conn: &Connection, table: &str) -> BTreeSet<String> {
    conn.prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))
        .and_then(|mut stmt| {
            stmt.query_map([], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<BTreeSet<String>>>()
        })
        .unwrap_or_default()
}

impl ShallowSessionProvider for OpencodeProvider {
    fn source(&self) -> &'static str {
        "opencode"
    }

    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        if !self.ensure_snapshot(env)? {
            return Ok(Vec::new());
        }
        let conn = self.open_snapshot()?;
        let columns = table_columns(&conn, "session");
        if !columns.contains("id") {
            return Ok(Vec::new());
        }
        let updated = if columns.contains("time_updated") {
            "time_updated"
        } else {
            "NULL"
        };
        let created = if columns.contains("time_created") {
            "time_created"
        } else {
            "NULL"
        };
        let sql = format!(
            "SELECT id, {created}, {updated} FROM session WHERE id IS NOT NULL AND id <> ''"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows
            .into_iter()
            .map(|(id, created, updated)| Candidate {
                source: "opencode",
                locator: id.clone(),
                session_id: Some(id),
                recency_hint_ms: updated.or(created),
                // Opencode has no file per session; the session's own
                // created/updated stamps are its change marker.
                stamp: format!("{}:{}", created.unwrap_or(0), updated.unwrap_or(0)),
            })
            .collect())
    }

    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        if !self.ensure_snapshot(env)? {
            return Ok(None);
        }
        let conn = self.open_snapshot()?;
        let columns = table_columns(&conn, "session");
        let directory = if columns.contains("directory") {
            "directory"
        } else {
            "NULL"
        };
        let created = if columns.contains("time_created") {
            "time_created"
        } else {
            "NULL"
        };
        let updated = if columns.contains("time_updated") {
            "time_updated"
        } else {
            "NULL"
        };
        let sql = format!("SELECT {directory}, {created}, {updated} FROM session WHERE id = ?");
        let row = conn
            .query_row(&sql, [&candidate.locator], |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            })
            .ok();
        let Some((directory, created, updated)) = row else {
            return Ok(None);
        };
        // The excerpt is cut in SQL, not in Rust: a single opencode part can
        // hold a whole pasted file, and materializing it just to take the
        // first 4096 characters would break the bounded-read promise for a
        // catalog entry.
        let first_prompt = conn
            .query_row(
                "SELECT substr(json_extract(p.data, '$.text'), 1, ?) \
                 FROM part p JOIN message m ON m.id = p.message_id \
                 WHERE p.session_id = ? AND json_extract(m.data, '$.role') = 'user' \
                 AND json_extract(p.data, '$.type') = 'text' \
                 ORDER BY COALESCE(p.time_created, m.time_created) ASC LIMIT 1",
                params![EXCERPT_MAX_CHARS as i64, &candidate.locator],
                |row| row.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten()
            .map(|text| excerpt(&text))
            .filter(|text| !text.is_empty());
        let mut models = Vec::new();
        if let Ok(model) = conn.query_row(
            "SELECT json_extract(data, '$.modelID') FROM message \
             WHERE session_id = ? AND json_extract(data, '$.modelID') IS NOT NULL LIMIT 1",
            [&candidate.locator],
            |row| row.get::<_, Option<String>>(0),
        ) {
            push_unique(&mut models, model.as_deref());
        }
        Ok(Some(ShallowSession {
            source: "opencode".into(),
            session_id: candidate.locator.clone(),
            cwd: directory,
            first_activity_ms: created,
            last_activity_ms: updated.or(created),
            first_prompt,
            models,
            ..Default::default()
        }))
    }
}

// ---------------------------------------------------------------------------
// relay
// ---------------------------------------------------------------------------

/// Shallow adapter for the network-backed `relay` source.
///
/// Relaycast has no local transcript files, and discovery must work with no
/// network access, so this adapter derives catalog rows from rows a previous
/// `ai-hist sync` already stored in `history` (indexed by
/// `idx_history_session`). If nothing was ever synced it discovers nothing —
/// that is the correct answer, not a failure.
struct RelayProvider;

impl ShallowSessionProvider for RelayProvider {
    fn source(&self) -> &'static str {
        "relay"
    }

    fn enumerate(&self, env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        let mut stmt = env.conn().prepare(
            "SELECT session_id, MAX(timestamp_ms), COUNT(*) FROM history \
             WHERE source = 'relay' AND session_id IS NOT NULL AND session_id <> '' \
             GROUP BY session_id",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows
            .into_iter()
            .map(|(session_id, last, count)| Candidate {
                source: "relay",
                locator: session_id.clone(),
                session_id: Some(session_id),
                recency_hint_ms: last,
                stamp: format!("{}:{count}", last.unwrap_or(0)),
            })
            .collect())
    }

    fn read_shallow(
        &self,
        env: &DiscoveryEnv<'_>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        let bounds = env.conn().query_row(
            "SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM history \
             WHERE source = 'relay' AND session_id = ?",
            [&candidate.locator],
            |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, Option<i64>>(1)?)),
        )?;
        let first_prompt = env
            .conn()
            .query_row(
                "SELECT prompt FROM history WHERE source = 'relay' AND session_id = ? \
                 ORDER BY timestamp_ms ASC, id ASC LIMIT 1",
                [&candidate.locator],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .map(|prompt| excerpt(&prompt))
            .filter(|prompt| !prompt.is_empty());
        Ok(Some(ShallowSession {
            source: "relay".into(),
            session_id: candidate.locator.clone(),
            first_activity_ms: bounds.0,
            last_activity_ms: bounds.1,
            first_prompt,
            ..Default::default()
        }))
    }
}

// ---------------------------------------------------------------------------
// shared enumeration helper
// ---------------------------------------------------------------------------

fn file_candidates(
    source: &'static str,
    files: Vec<PathBuf>,
    stamp: fn(&Path) -> Result<String>,
) -> Result<Vec<Candidate>> {
    let mut out = Vec::with_capacity(files.len());
    for path in files {
        // A file that vanished between the walk and the stat is not an error;
        // the next run will simply not see it.
        let Ok(stamp) = stamp(&path) else { continue };
        out.push(Candidate {
            source,
            locator: path.to_string_lossy().into_owned(),
            session_id: None,
            recency_hint_ms: crate::file_modified_ms(&path),
            stamp,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// catalog reads and writes
// ---------------------------------------------------------------------------

const SESSION_COLUMNS: &str = "source, session_id, cwd, git_branch, first_activity_ms, \
     last_activity_ms, first_prompt, last_assistant_text, models_json, originator, \
     agent_version, repo_url, initial_commit, workspace_roots_json, raw_path, source_stamp, \
     discovery_state";

fn json_string_list(raw: Option<String>) -> Vec<String> {
    raw.and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
        .unwrap_or_default()
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<ShallowSession> {
    Ok(ShallowSession {
        source: row.get(0)?,
        session_id: row.get(1)?,
        cwd: row.get(2)?,
        git_branch: row.get(3)?,
        first_activity_ms: row.get(4)?,
        last_activity_ms: row.get(5)?,
        first_prompt: row.get(6)?,
        last_assistant_text: row.get(7)?,
        models: json_string_list(row.get(8)?),
        originator: row.get(9)?,
        agent_version: row.get(10)?,
        repo_url: row.get(11)?,
        initial_commit: row.get(12)?,
        workspace_roots: json_string_list(row.get(13)?),
        raw_path: row.get(14)?,
        source_stamp: row.get(15)?,
        discovery_state: row
            .get::<_, Option<String>>(16)?
            .unwrap_or_else(|| "full".to_string()),
        from_cache: true,
    })
}

/// A precise continuation point in the catalog's total order.
///
/// The catalog is ordered `(last_activity_ms DESC, source ASC, session_id ASC)`.
/// Recency alone is not a key: a single discovery pass can stamp dozens of
/// sessions with the same mtime-derived millisecond, and a cursor that carries
/// only a timestamp silently drops every row tied with the page boundary. The
/// identity columns make the cursor total.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CatalogCursor {
    /// Last activity of the final row on the previous page; `None` for a row
    /// whose recency is unknown (those sort last, after every dated row).
    pub last_activity_ms: Option<i64>,
    /// Source of the final row on the previous page.
    pub source: String,
    /// Session id of the final row on the previous page.
    pub session_id: String,
}

/// Options for the cache-only catalog listing.
#[derive(Debug, Clone, Default)]
pub struct CatalogListOptions {
    /// Restrict to these sources. Empty means every discoverable source.
    pub sources: Vec<String>,
    /// Row cap; defaults to [`DEFAULT_CATALOG_LIMIT`].
    pub limit: Option<i64>,
    /// Coarse cutoff: only sessions strictly older than this millisecond.
    /// Convenient for "show me anything before last Tuesday", but it cannot
    /// separate rows that share a millisecond — use [`CatalogListOptions::after`]
    /// to walk pages. Ignored when `after` is set.
    pub before_ms: Option<i64>,
    /// Precise continuation from the previous page's `next_cursor`.
    pub after: Option<CatalogCursor>,
}

/// One page of the catalog plus the cursor that continues it.
#[derive(Debug, Clone, Default)]
pub struct SessionCatalogPage {
    /// The rows, newest first.
    pub sessions: Vec<ShallowSession>,
    /// Pass as [`CatalogListOptions::after`] for the next page. `None` when
    /// this page did not fill its limit, i.e. the catalog is exhausted.
    pub next_cursor: Option<CatalogCursor>,
}

/// The catalog listing query and its bound arguments.
///
/// Built in one place so the query-plan test asserts the plan of the statement
/// that actually runs.
fn catalog_list_query(options: &CatalogListOptions) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
    let mut sql = format!("SELECT {SESSION_COLUMNS} FROM sessions WHERE source <> 'trajectory'");
    let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if !options.sources.is_empty() {
        let placeholders = vec!["?"; options.sources.len()].join(", ");
        sql.push_str(&format!(" AND source IN ({placeholders})"));
        for source in &options.sources {
            args.push(Box::new(source.clone()));
        }
    }
    match options.after.as_ref() {
        // Everything strictly after the cursor in the catalog's total order.
        // Undated rows sort last, so a dated cursor must still reach them.
        Some(cursor) => match cursor.last_activity_ms {
            Some(ms) => {
                sql.push_str(
                    " AND (last_activity_ms IS NULL OR last_activity_ms < ? \
                       OR (last_activity_ms = ? \
                           AND (source > ? OR (source = ? AND session_id > ?))))",
                );
                args.push(Box::new(ms));
                args.push(Box::new(ms));
                args.push(Box::new(cursor.source.clone()));
                args.push(Box::new(cursor.source.clone()));
                args.push(Box::new(cursor.session_id.clone()));
            }
            None => {
                sql.push_str(
                    " AND last_activity_ms IS NULL \
                       AND (source > ? OR (source = ? AND session_id > ?))",
                );
                args.push(Box::new(cursor.source.clone()));
                args.push(Box::new(cursor.source.clone()));
                args.push(Box::new(cursor.session_id.clone()));
            }
        },
        None => {
            if let Some(before_ms) = options.before_ms {
                sql.push_str(" AND last_activity_ms < ?");
                args.push(Box::new(before_ms));
            }
        }
    }
    sql.push_str(" ORDER BY last_activity_ms DESC, source ASC, session_id ASC LIMIT ?");
    args.push(Box::new(options.limit.unwrap_or(DEFAULT_CATALOG_LIMIT)));
    (sql, args)
}

/// List the session catalog straight out of the database.
///
/// Pure SQL over `sessions`: no filesystem access, no provider I/O, and no
/// scan of `history` / `session_events` / `tool_calls`. `trajectory` rows are
/// excluded defensively — trajectories are derived records, not sessions, and
/// must never appear in a session list even if something wrote one.
///
/// Rows come back in the catalog's total order:
/// `(last_activity_ms DESC, source ASC, session_id ASC)`, with rows of unknown
/// recency last. Use [`list_session_catalog_page`] to paginate.
pub fn list_session_catalog(
    conn: &Connection,
    options: &CatalogListOptions,
) -> Result<Vec<ShallowSession>> {
    let (sql, args) = catalog_list_query(options);
    let mut stmt = conn.prepare(&sql)?;
    let params = rusqlite::params_from_iter(args.iter().map(|arg| arg.as_ref()));
    let rows = stmt
        .query_map(params, row_to_session)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// [`list_session_catalog`] plus the cursor that continues it.
///
/// The cursor is `None` once a page comes back short of its limit, so a
/// caller walks the catalog by following `next_cursor` until it is absent —
/// no duplicated and no skipped rows, even when a whole page shares one
/// millisecond.
pub fn list_session_catalog_page(
    conn: &Connection,
    options: &CatalogListOptions,
) -> Result<SessionCatalogPage> {
    let sessions = list_session_catalog(conn, options)?;
    let limit = options.limit.unwrap_or(DEFAULT_CATALOG_LIMIT);
    let next_cursor = (limit > 0 && sessions.len() as i64 >= limit)
        .then(|| sessions.last())
        .flatten()
        .map(|row| CatalogCursor {
            last_activity_ms: row.last_activity_ms,
            source: row.source.clone(),
            session_id: row.session_id.clone(),
        });
    Ok(SessionCatalogPage {
        sessions,
        next_cursor,
    })
}

fn fetch_catalog_row(
    conn: &Connection,
    source: &str,
    session_id: &str,
) -> Result<Option<ShallowSession>> {
    let sql = format!("SELECT {SESSION_COLUMNS} FROM sessions WHERE source = ? AND session_id = ?");
    Ok(conn
        .query_row(&sql, params![source, session_id], row_to_session)
        .ok())
}

fn fetch_catalog_row_by_path(
    conn: &Connection,
    source: &str,
    raw_path: &str,
) -> Result<Option<ShallowSession>> {
    let sql =
        format!("SELECT {SESSION_COLUMNS} FROM sessions WHERE source = ? AND raw_path = ? LIMIT 1");
    Ok(conn
        .query_row(&sql, params![source, raw_path], row_to_session)
        .ok())
}

/// Whether this source was already examined at this exact stamp and found not
/// to be a session.
///
/// Without this, every codex subagent thread and every claude sidecar — real
/// files that legitimately produce no catalog row — was re-read on every
/// single run, because "no row" left nothing for the stamp check to match.
fn is_known_non_session(
    conn: &Connection,
    source: &str,
    locator: &str,
    stamp: &str,
) -> Result<bool> {
    let known: Option<String> = conn
        .query_row(
            "SELECT stamp FROM discovery_skips WHERE source = ? AND locator = ?",
            params![source, locator],
            |row| row.get(0),
        )
        .ok();
    Ok(known.as_deref() == Some(stamp))
}

/// Remember that this source, at this stamp, is not a session.
fn record_non_session(conn: &Connection, source: &str, locator: &str, stamp: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO discovery_skips (source, locator, stamp, reason, updated_ms) \
         VALUES (?, ?, ?, 'not-a-session', ?) \
         ON CONFLICT(source, locator) DO UPDATE SET \
         stamp = excluded.stamp, reason = excluded.reason, updated_ms = excluded.updated_ms",
        params![source, locator, stamp, now_ms()],
    )?;
    Ok(())
}

/// Drop a stale non-session marker once a source does resolve to a session.
fn clear_non_session(conn: &Connection, source: &str, locator: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM discovery_skips WHERE source = ? AND locator = ?",
        params![source, locator],
    )?;
    Ok(())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or_default()
}

/// Write a shallow row into the catalog.
///
/// Never nulls out a value the catalog already holds, never lowers
/// `first_activity_ms` past what a fuller pass observed, and never downgrades
/// a fully indexed row to `'shallow'` — including a row from a database that
/// predates `discovery_state`, whose NULL readers deliberately interpret as
/// `'full'`. A shallow rescan of such a row still refreshes its metadata and
/// stamp.
pub fn upsert_shallow_session(conn: &Connection, session: &ShallowSession) -> Result<()> {
    conn.execute(
        "INSERT INTO sessions \
         (session_id, source, cwd, git_branch, first_activity_ms, last_activity_ms, \
          last_assistant_text, raw_path, parser_version, first_prompt, models_json, originator, \
          agent_version, repo_url, initial_commit, workspace_roots_json, source_stamp, \
          discovery_state) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, 'shallow') \
         ON CONFLICT(session_id, source) DO UPDATE SET \
         cwd = COALESCE(excluded.cwd, sessions.cwd), \
         git_branch = COALESCE(excluded.git_branch, sessions.git_branch), \
         first_activity_ms = CASE \
             WHEN excluded.first_activity_ms IS NULL THEN sessions.first_activity_ms \
             WHEN sessions.first_activity_ms IS NULL THEN excluded.first_activity_ms \
             ELSE MIN(sessions.first_activity_ms, excluded.first_activity_ms) END, \
         last_activity_ms = CASE \
             WHEN excluded.last_activity_ms IS NULL THEN sessions.last_activity_ms \
             WHEN sessions.last_activity_ms IS NULL THEN excluded.last_activity_ms \
             ELSE MAX(sessions.last_activity_ms, excluded.last_activity_ms) END, \
         last_assistant_text = COALESCE(excluded.last_assistant_text, sessions.last_assistant_text), \
         raw_path = COALESCE(excluded.raw_path, sessions.raw_path), \
         first_prompt = COALESCE(excluded.first_prompt, sessions.first_prompt), \
         models_json = COALESCE(excluded.models_json, sessions.models_json), \
         originator = COALESCE(excluded.originator, sessions.originator), \
         agent_version = COALESCE(excluded.agent_version, sessions.agent_version), \
         repo_url = COALESCE(excluded.repo_url, sessions.repo_url), \
         initial_commit = COALESCE(excluded.initial_commit, sessions.initial_commit), \
         workspace_roots_json = COALESCE(excluded.workspace_roots_json, sessions.workspace_roots_json), \
         source_stamp = COALESCE(excluded.source_stamp, sessions.source_stamp), \
         discovery_state = CASE \
             WHEN sessions.discovery_state IS NULL OR sessions.discovery_state = 'full' \
             THEN 'full' ELSE 'shallow' END",
        params![
            session.session_id,
            session.source,
            session.cwd,
            session.git_branch,
            session.first_activity_ms,
            session.last_activity_ms,
            session.last_assistant_text,
            session.raw_path,
            session.first_prompt,
            json_array_or_none(&session.models),
            session.originator,
            session.agent_version,
            session.repo_url,
            session.initial_commit,
            json_array_or_none(&session.workspace_roots),
            session.source_stamp,
        ],
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// discovery engine
// ---------------------------------------------------------------------------

/// Options for one discovery run.
#[derive(Debug, Clone, Default)]
pub struct DiscoverOptions {
    /// Restrict to these sources. Empty means every adapter.
    pub sources: Vec<String>,
    /// Global cap on emitted rows, applied across providers by recency.
    /// `None` means no cap.
    pub limit: Option<usize>,
}

/// Something one provider (or one session) could not do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveryDiagnostic {
    /// Source the failure belongs to.
    pub source: String,
    /// Candidate locator, when the failure was scoped to one session.
    pub locator: Option<String>,
    /// Human-readable cause.
    pub error: String,
}

/// Per-provider tallies for one run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ProviderSummary {
    /// Candidates this provider enumerated (before the global limit).
    pub candidates: usize,
    /// Rows emitted after a shallow read.
    pub discovered: usize,
    /// Rows served from the catalog because the stamp was unchanged.
    pub skipped_unchanged: usize,
    /// `true` when enumeration itself failed.
    pub failed: bool,
}

/// Outcome of one discovery run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DiscoverySummary {
    /// [`SESSION_CATALOG_CONTRACT_VERSION`].
    pub contract_version: u32,
    /// Rows freshly read and upserted.
    pub discovered: usize,
    /// Rows served from the catalog on an unchanged stamp.
    pub skipped_unchanged: usize,
    /// Per-provider tallies, keyed by source.
    pub providers: BTreeMap<String, ProviderSummary>,
    /// Sources that deliberately have no adapter.
    pub exempt_sources: Vec<SourceExemption>,
    /// Non-fatal failures. A provider failing here never blocks another.
    pub diagnostics: Vec<DiscoveryDiagnostic>,
    /// Work actually performed.
    pub counters: DiscoveryCounters,
}

/// Every selected provider failed to enumerate, so the run made no progress.
///
/// Carries the run's summary — diagnostics included — so a caller can report
/// what each provider said before propagating the failure.
#[derive(Debug)]
pub struct AllProvidersFailed {
    /// The run as far as it got, including one diagnostic per failed provider.
    pub summary: DiscoverySummary,
}

impl std::fmt::Display for AllProvidersFailed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "all {} session provider(s) failed; no provider made progress",
            self.summary
                .providers
                .values()
                .filter(|provider| provider.failed)
                .count()
        )
    }
}

impl std::error::Error for AllProvidersFailed {}

fn stored_stamp(raw: &str) -> String {
    format!("v{SHALLOW_SCANNER_VERSION}:{raw}")
}

fn select_providers(options: &DiscoverOptions) -> Result<Vec<Box<dyn ShallowSessionProvider>>> {
    for source in &options.sources {
        if let Some(exempt) = DISCOVERY_EXEMPTIONS
            .iter()
            .find(|entry| entry.source == source)
        {
            anyhow::bail!(
                "source '{}' is exempt from session discovery: {}",
                exempt.source,
                exempt.reason
            );
        }
        anyhow::ensure!(
            SOURCE_CHOICES.contains(&source.as_str()),
            "invalid source '{source}' (choose from {})",
            SOURCE_CHOICES.join(", ")
        );
    }
    let mut providers = shallow_providers();
    if !options.sources.is_empty() {
        providers.retain(|provider| options.sources.iter().any(|s| s == provider.source()));
    }
    Ok(providers)
}

/// Discover sessions across every provider, newest first, emitting rows as
/// they are produced.
///
/// The ordering and the limit are **global**: candidates from all providers
/// are merged by recency hint and only then truncated, so `--limit 3` over two
/// providers returns the three newest sessions overall, not three from
/// whichever provider was enumerated first.
///
/// A provider whose enumeration fails contributes a diagnostic and nothing
/// else; the rest of the run continues. A malformed or unreadable individual
/// session likewise yields a per-session diagnostic. The call only fails when
/// *every* selected provider failed.
pub fn discover_sessions(
    conn: &Connection,
    options: &DiscoverOptions,
    on_row: impl FnMut(&ShallowSession),
) -> Result<DiscoverySummary> {
    let env = DiscoveryEnv::new(conn);
    discover_sessions_with_env(&env, options, on_row)
}

/// [`discover_sessions`] against an explicitly built [`DiscoveryEnv`].
pub fn discover_sessions_with_env(
    env: &DiscoveryEnv<'_>,
    options: &DiscoverOptions,
    mut on_row: impl FnMut(&ShallowSession),
) -> Result<DiscoverySummary> {
    let conn = env.conn();
    let providers = select_providers(options)?;
    let mut summary = DiscoverySummary {
        contract_version: SESSION_CATALOG_CONTRACT_VERSION,
        exempt_sources: DISCOVERY_EXEMPTIONS.to_vec(),
        ..Default::default()
    };

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut failed_providers = 0usize;
    for provider in &providers {
        let entry = summary
            .providers
            .entry(provider.source().to_string())
            .or_default();
        match provider.enumerate(env) {
            Ok(found) => {
                entry.candidates = found.len();
                env.note_candidates(found.len() as u64);
                candidates.extend(found);
            }
            Err(error) => {
                entry.failed = true;
                failed_providers += 1;
                summary.diagnostics.push(DiscoveryDiagnostic {
                    source: provider.source().to_string(),
                    locator: None,
                    error: format!("{error:#}"),
                });
            }
        }
    }
    if !providers.is_empty() && failed_providers == providers.len() {
        // Still a failure, but the diagnostics explaining *why* each provider
        // failed are the useful part. Carrying the summary inside the error
        // lets a JSONL consumer receive the diagnostic lines and a summary
        // trailer before the non-zero exit, instead of a bare message.
        summary.counters = env.counters();
        return Err(AllProvidersFailed { summary }.into());
    }

    // Global recency ordering. Candidates with no recency signal sort last;
    // ties break on (source, locator) so a run is reproducible.
    candidates.sort_by(|a, b| {
        b.recency_hint_ms
            .cmp(&a.recency_hint_ms)
            .then_with(|| a.source.cmp(b.source))
            .then_with(|| a.locator.cmp(&b.locator))
    });
    let by_source: BTreeMap<&str, &dyn ShallowSessionProvider> = providers
        .iter()
        .map(|provider| (provider.source(), provider.as_ref()))
        .collect();

    // The limit counts *emitted sessions*, not candidates. Truncating the
    // candidate list up front let a codex subagent thread or a claude sidecar
    // -- neither of which is a session -- eat a result slot, so `--limit 3`
    // could hand back two sessions while older valid ones went unread.
    let limit = options.limit.unwrap_or(usize::MAX);
    let mut emitted = 0usize;
    // One session can be reached through more than one file in a single run
    // (a transcript plus its subagent sidecars). Emit it once.
    let mut emitted_sessions: BTreeSet<(String, String)> = BTreeSet::new();

    for candidate in &candidates {
        if emitted >= limit {
            break;
        }
        let Some(provider) = by_source.get(candidate.source) else {
            continue;
        };
        let expected = stored_stamp(&candidate.stamp);
        let cached = match candidate.session_id.as_deref() {
            Some(session_id) => fetch_catalog_row(conn, candidate.source, session_id)?,
            None => fetch_catalog_row_by_path(conn, candidate.source, &candidate.locator)?,
        };
        if let Some(cached) = cached.filter(|row| row.source_stamp.as_deref() == Some(&expected)) {
            env.note_skipped();
            summary.skipped_unchanged += 1;
            if let Some(entry) = summary.providers.get_mut(candidate.source) {
                entry.skipped_unchanged += 1;
            }
            if emitted_sessions.insert((cached.source.clone(), cached.session_id.clone())) {
                emitted += 1;
                on_row(&cached);
            }
            continue;
        }
        // A source already examined and found not to be a session (a codex
        // subagent thread, a claude sidecar) is remembered by its stamp, so a
        // rescan costs a PK lookup instead of a fresh read every single run.
        if is_known_non_session(conn, candidate.source, &candidate.locator, &expected)? {
            env.note_skipped();
            summary.skipped_unchanged += 1;
            if let Some(entry) = summary.providers.get_mut(candidate.source) {
                entry.skipped_unchanged += 1;
            }
            continue;
        }
        env.note_shallow_read();
        let read = provider.read_shallow(env, candidate);
        let session = match read {
            Ok(Some(session)) => session,
            Ok(None) => {
                record_non_session(conn, candidate.source, &candidate.locator, &expected)?;
                continue;
            }
            Err(error) => {
                summary.diagnostics.push(DiscoveryDiagnostic {
                    source: candidate.source.to_string(),
                    locator: Some(candidate.locator.clone()),
                    error: format!("{error:#}"),
                });
                continue;
            }
        };
        if session.session_id.is_empty() {
            summary.diagnostics.push(DiscoveryDiagnostic {
                source: candidate.source.to_string(),
                locator: Some(candidate.locator.clone()),
                error: "no session id in source".to_string(),
            });
            continue;
        }
        let mut session = session;
        session.source_stamp = Some(expected);
        session.discovery_state = "shallow".to_string();
        if let Err(error) = upsert_shallow_session(conn, &session) {
            summary.diagnostics.push(DiscoveryDiagnostic {
                source: candidate.source.to_string(),
                locator: Some(candidate.locator.clone()),
                error: format!("{error:#}"),
            });
            continue;
        }
        // A file that used to be skipped as a non-session (or was never one)
        // must not keep a stale marker once it resolves to a session.
        clear_non_session(conn, candidate.source, &candidate.locator)?;
        // Emit the merged catalog row, so what a caller sees is exactly what
        // the catalog now holds (including a preserved `full` state).
        let row = fetch_catalog_row(conn, &session.source, &session.session_id)?
            .map(|mut row| {
                row.from_cache = false;
                row
            })
            .unwrap_or(session);
        summary.discovered += 1;
        if let Some(entry) = summary.providers.get_mut(candidate.source) {
            entry.discovered += 1;
        }
        if emitted_sessions.insert((row.source.clone(), row.session_id.clone())) {
            emitted += 1;
            on_row(&row);
        }
    }

    summary.counters = env.counters();
    Ok(summary)
}

/// [`discover_sessions`] with the rows collected instead of streamed.
pub fn discover_sessions_collect(
    conn: &Connection,
    options: &DiscoverOptions,
) -> Result<(Vec<ShallowSession>, DiscoverySummary)> {
    let mut rows = Vec::new();
    let summary = discover_sessions(conn, options, |session| rows.push(session.clone()))?;
    Ok((rows, summary))
}

#[cfg(test)]
mod tests;
