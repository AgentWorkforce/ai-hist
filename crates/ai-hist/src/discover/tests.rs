//! Fixture-backed tests for shallow session discovery.
//!
//! Every fixture is written inline into a temp directory and reached through an
//! explicit [`DiscoveryEnv`], so no test mutates process-wide environment
//! variables and the suite stays parallel-safe.
//!
//! Performance claims are asserted through [`DiscoveryCounters`] rather than a
//! wall clock: bounded reads mean a limited request opens a bounded number of
//! files and reads a bounded number of bytes no matter how large the archive
//! is, an unchanged rescan performs zero shallow reads, and the cache-only
//! listing performs no file I/O at all.

use super::*;
use ai_hist_core::init_db;
use std::fs;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn catalog() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory database");
    init_db(&conn).expect("schema");
    conn
}

fn env_at<'a>(conn: &'a Connection, home: &Path) -> DiscoveryEnv<'a> {
    DiscoveryEnv::with_roots(conn, home.to_path_buf(), home.join("opencode.db"))
}

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, contents).expect("write fixture");
}

/// Pin a file's mtime so recency ordering (and the change stamp) is
/// deterministic instead of depending on how fast the test ran.
fn set_mtime(path: &Path, ms: i64) {
    let file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open for mtime");
    let when = SystemTime::UNIX_EPOCH + Duration::from_millis(ms as u64);
    file.set_times(fs::FileTimes::new().set_modified(when))
        .expect("set mtime");
}

fn claude_session(home: &Path, id: &str, body: &str, mtime_ms: i64) -> PathBuf {
    let path = home.join(format!(".claude/projects/proj/{id}.jsonl"));
    write(&path, body);
    set_mtime(&path, mtime_ms);
    path
}

fn codex_rollout(home: &Path, id: &str, body: &str, mtime_ms: i64) -> PathBuf {
    let path = home.join(format!(".codex/sessions/2026/06/20/rollout-{id}.jsonl"));
    write(&path, body);
    set_mtime(&path, mtime_ms);
    path
}

fn cursor_session(home: &Path, project: &str, id: &str, body: &str, mtime_ms: i64) -> PathBuf {
    let path = home.join(format!(
        ".cursor/projects/{project}/agent-transcripts/{id}/{id}.jsonl"
    ));
    write(&path, body);
    set_mtime(&path, mtime_ms);
    path
}

fn grok_session(home: &Path, project: &str, id: &str, summary: &str, chat: &str, mtime_ms: i64) {
    let dir = home.join(format!(".grok/sessions/{project}/{id}"));
    write(&dir.join("summary.json"), summary);
    write(&dir.join("chat_history.jsonl"), chat);
    set_mtime(&dir.join("summary.json"), mtime_ms);
    set_mtime(&dir.join("chat_history.jsonl"), mtime_ms);
}

fn opencode_db(home: &Path, statements: &str) {
    fs::create_dir_all(home).expect("mkdir");
    let db = Connection::open(home.join("opencode.db")).expect("opencode db");
    db.execute_batch(&format!(
        "CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT, time_created INTEGER, time_updated INTEGER);
         CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
         CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);
         {statements}"
    ))
    .expect("opencode fixture");
}

/// A realistic Claude transcript: a meta row, a slash-command wrapper, and a
/// sidechain (subagent) turn all precede the first real human prompt.
const CLAUDE_BODY: &str = concat!(
    r#"{"sessionId":"claude-1","cwd":"/work/app","gitBranch":"main","version":"1.2.3","type":"user","isMeta":true,"message":{"role":"user","content":"session bookkeeping"},"timestamp":"2026-06-20T10:00:00.000Z"}"#,
    "\n",
    r#"{"sessionId":"claude-1","type":"user","message":{"role":"user","content":"<command-name>/compact</command-name>"},"timestamp":"2026-06-20T10:00:01.000Z"}"#,
    "\n",
    r#"{"sessionId":"claude-1","type":"user","isSidechain":true,"message":{"role":"user","content":"subagent instruction"},"timestamp":"2026-06-20T10:00:02.000Z"}"#,
    "\n",
    r#"{"sessionId":"claude-1","type":"user","message":{"role":"user","content":[{"type":"text","text":"the real first prompt"}]},"timestamp":"2026-06-20T10:00:03.000Z"}"#,
    "\n",
    r#"{"sessionId":"claude-1","type":"assistant","message":{"role":"assistant","model":"claude-opus-4","content":[{"type":"text","text":"working on it"}]},"timestamp":"2026-06-20T10:05:00.000Z"}"#,
    "\n",
);

const CODEX_BODY: &str = concat!(
    r#"{"timestamp":"2026-06-20T11:00:00.000Z","type":"session_meta","payload":{"id":"codex-1","cwd":"/work/api","originator":"codex_cli_rs","cli_version":"0.148.0","git":{"branch":"feature","repository_url":"git@github.com:acme/api.git","commit_hash":"abc1234"}}}"#,
    "\n",
    r#"{"timestamp":"2026-06-20T11:00:01.000Z","type":"turn_context","payload":{"model":"gpt-5-codex"}}"#,
    "\n",
    r#"{"timestamp":"2026-06-20T11:00:02.000Z","type":"event_msg","payload":{"type":"user_message","message":"<environment_context>ignore me</environment_context>"}}"#,
    "\n",
    r#"{"timestamp":"2026-06-20T11:00:03.000Z","type":"event_msg","payload":{"type":"user_message","message":"add a retry to the client"}}"#,
    "\n",
    r#"{"timestamp":"2026-06-20T11:09:00.000Z","type":"event_msg","payload":{"type":"agent_message","message":"done"}}"#,
    "\n",
);

fn discover(conn: &Connection, home: &Path, options: &DiscoverOptions) -> DiscoveryResultForTest {
    let env = env_at(conn, home);
    let mut rows = Vec::new();
    let summary =
        discover_sessions_with_env(&env, options, |session| rows.push(session.clone())).unwrap();
    DiscoveryResultForTest { rows, summary }
}

struct DiscoveryResultForTest {
    rows: Vec<ShallowSession>,
    summary: DiscoverySummary,
}

impl DiscoveryResultForTest {
    fn ids(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| format!("{}:{}", row.source, row.session_id))
            .collect()
    }

    fn row(&self, session_id: &str) -> &ShallowSession {
        self.rows
            .iter()
            .find(|row| row.session_id == session_id)
            .unwrap_or_else(|| panic!("no {session_id} in {:?}", self.ids()))
    }
}

fn only(sources: &[&str]) -> DiscoverOptions {
    DiscoverOptions {
        sources: sources.iter().map(|s| s.to_string()).collect(),
        limit: None,
    }
}

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

#[test]
fn every_source_is_either_discoverable_or_explicitly_exempt() {
    let adapters: BTreeSet<&str> = shallow_providers()
        .iter()
        .map(|provider| provider.source())
        .collect();
    let exempt: BTreeSet<&str> = DISCOVERY_EXEMPTIONS
        .iter()
        .map(|entry| entry.source)
        .collect();
    assert_eq!(
        adapters.len(),
        shallow_providers().len(),
        "an adapter is registered twice"
    );
    for source in SOURCE_CHOICES {
        let covered = usize::from(adapters.contains(source)) + usize::from(exempt.contains(source));
        assert_eq!(
            covered, 1,
            "source '{source}' must be covered by exactly one of the adapter or exemption lists; \
             a new provider needs a decision, not a default"
        );
    }
    for source in adapters.iter().chain(exempt.iter()) {
        assert!(
            SOURCE_CHOICES.contains(source),
            "'{source}' is registered but is not a known source"
        );
    }
    assert!(
        exempt.contains("trajectory"),
        "trajectories are derived records and must stay out of session discovery"
    );
}

#[test]
fn discovering_an_exempt_source_is_rejected_with_its_reason() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    let env = env_at(&conn, home.path());
    let error = discover_sessions_with_env(&env, &only(&["trajectory"]), |_| {}).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("exempt"), "{message}");
    assert!(message.contains("derived trajectory records"), "{message}");
}

#[test]
fn discovering_an_unknown_source_is_rejected() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    let env = env_at(&conn, home.path());
    let error = discover_sessions_with_env(&env, &only(&["nope"]), |_| {}).unwrap_err();
    assert!(format!("{error:#}").contains("invalid source"));
}

// ---------------------------------------------------------------------------
// claude
// ---------------------------------------------------------------------------

#[test]
fn claude_cold_discovery_extracts_identity_and_skips_non_human_turns() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);

    let found = discover(&conn, home.path(), &only(&["claude"]));
    assert_eq!(found.ids(), vec!["claude:claude-1"]);
    let row = found.row("claude-1");
    assert_eq!(row.cwd.as_deref(), Some("/work/app"));
    assert_eq!(row.git_branch.as_deref(), Some("main"));
    assert_eq!(row.agent_version.as_deref(), Some("1.2.3"));
    assert_eq!(row.models, vec!["claude-opus-4".to_string()]);
    assert_eq!(row.discovery_state, "shallow");
    assert!(!row.from_cache);
    // Meta, slash-command wrapper and sidechain turns are not human prompts.
    assert_eq!(row.first_prompt.as_deref(), Some("the real first prompt"));
    assert_eq!(
        row.first_activity_ms,
        crate::parse_iso_ms("2026-06-20T10:00:00.000Z")
    );
    assert_eq!(
        row.last_activity_ms,
        crate::parse_iso_ms("2026-06-20T10:05:00.000Z")
    );
    assert_eq!(found.summary.discovered, 1);
    assert_eq!(
        found.summary.contract_version,
        SESSION_CATALOG_CONTRACT_VERSION
    );
}

#[test]
fn claude_identity_is_stable_and_an_unchanged_rescan_reparses_nothing() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);

    let first = discover(&conn, home.path(), &only(&["claude"]));
    assert_eq!(first.summary.counters.shallow_reads, 1);

    let second = discover(&conn, home.path(), &only(&["claude"]));
    assert_eq!(second.ids(), first.ids(), "session identity must be stable");
    assert_eq!(
        second.summary.counters.shallow_reads, 0,
        "an unchanged stamp must skip the read entirely"
    );
    assert_eq!(second.summary.skipped_unchanged, 1);
    assert_eq!(second.summary.counters.files_opened, 0);
    assert!(second.rows[0].from_cache);

    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 1, "a rescan must upsert, never duplicate");
}

#[test]
fn claude_append_updates_the_existing_row_without_duplicating_it() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    let path = claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);
    discover(&conn, home.path(), &only(&["claude"]));

    let mut appended = fs::read_to_string(&path).unwrap();
    appended.push_str(
        r#"{"sessionId":"claude-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"later"}]},"timestamp":"2026-06-20T12:00:00.000Z"}"#,
    );
    appended.push('\n');
    fs::write(&path, appended).unwrap();
    set_mtime(&path, 1_750_000_500_000);

    let second = discover(&conn, home.path(), &only(&["claude"]));
    assert_eq!(second.summary.counters.shallow_reads, 1);
    assert_eq!(
        second.row("claude-1").last_activity_ms,
        crate::parse_iso_ms("2026-06-20T12:00:00.000Z")
    );
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 1);
}

#[test]
fn an_incomplete_trailing_record_is_ignored_but_the_session_still_appears() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    // A transcript being written right now: the final record has no newline.
    let body = format!(
        "{CLAUDE_BODY}{}",
        r#"{"sessionId":"claude-1","type":"user","message":{"role":"user","content":"half-written"#
    );
    claude_session(home.path(), "claude-1", &body, 1_750_000_000_000);

    let found = discover(&conn, home.path(), &only(&["claude"]));
    assert_eq!(found.ids(), vec!["claude:claude-1"]);
    assert_eq!(
        found.row("claude-1").last_activity_ms,
        crate::parse_iso_ms("2026-06-20T10:05:00.000Z"),
        "a partial trailing record is not yet a record"
    );
}

#[test]
fn a_malformed_transcript_does_not_hide_its_healthy_neighbours() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);
    // Not JSON at all, and no session id anywhere.
    claude_session(
        home.path(),
        "broken",
        "}}}} not json at all\nalso not json\n",
        1_750_000_100_000,
    );

    let found = discover(&conn, home.path(), &only(&["claude"]));
    let ids = found.ids();
    assert!(ids.contains(&"claude:claude-1".to_string()), "{ids:?}");
    // The unparseable file still has a stable identity from its file name, but
    // it must never take the healthy session's row with it.
    assert_eq!(found.summary.providers["claude"].candidates, 2);
    assert!(found.summary.discovered >= 1);
}

#[test]
fn absent_optional_metadata_stays_null_instead_of_being_invented() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(
        home.path(),
        "bare",
        concat!(
            r#"{"sessionId":"bare","type":"user","message":{"role":"user","content":"hello"}}"#,
            "\n"
        ),
        1_750_000_000_000,
    );

    let found = discover(&conn, home.path(), &only(&["claude"]));
    let row = found.row("bare");
    assert_eq!(row.cwd, None);
    assert_eq!(row.git_branch, None);
    assert_eq!(row.agent_version, None);
    assert_eq!(row.repo_url, None);
    assert_eq!(row.initial_commit, None);
    assert_eq!(row.originator, None);
    assert_eq!(row.first_activity_ms, None);
    assert_eq!(row.last_activity_ms, None);
    assert!(row.models.is_empty());
    assert!(row.workspace_roots.is_empty());
    // The columns really are NULL, not empty JSON arrays.
    let (models, roots): (Option<String>, Option<String>) = conn
        .query_row(
            "SELECT models_json, workspace_roots_json FROM sessions WHERE session_id = 'bare'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(models, None);
    assert_eq!(roots, None);
}

// ---------------------------------------------------------------------------
// codex
// ---------------------------------------------------------------------------

#[test]
fn codex_session_meta_supplies_originator_version_and_git_provenance() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    codex_rollout(home.path(), "codex-1", CODEX_BODY, 1_750_000_200_000);

    let found = discover(&conn, home.path(), &only(&["codex"]));
    let row = found.row("codex-1");
    assert_eq!(row.cwd.as_deref(), Some("/work/api"));
    assert_eq!(row.git_branch.as_deref(), Some("feature"));
    assert_eq!(row.originator.as_deref(), Some("codex_cli_rs"));
    assert_eq!(row.agent_version.as_deref(), Some("0.148.0"));
    assert_eq!(row.repo_url.as_deref(), Some("git@github.com:acme/api.git"));
    assert_eq!(row.initial_commit.as_deref(), Some("abc1234"));
    assert_eq!(row.models, vec!["gpt-5-codex".to_string()]);
    assert_eq!(
        row.first_prompt.as_deref(),
        Some("add a retry to the client"),
        "the environment_context control turn is not a human prompt"
    );
    assert_eq!(
        row.last_activity_ms,
        crate::parse_iso_ms("2026-06-20T11:09:00.000Z")
    );
}

#[test]
fn codex_subagent_threads_are_not_sessions() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    codex_rollout(home.path(), "codex-1", CODEX_BODY, 1_750_000_200_000);
    codex_rollout(
        home.path(),
        "codex-sub",
        concat!(
            r#"{"timestamp":"2026-06-20T11:02:00.000Z","type":"session_meta","payload":{"id":"codex-sub","cwd":"/work/api","thread_source":"subagent"}}"#,
            "\n"
        ),
        1_750_000_300_000,
    );

    let found = discover(&conn, home.path(), &only(&["codex"]));
    assert_eq!(found.ids(), vec!["codex:codex-1"]);
    assert_eq!(found.summary.providers["codex"].candidates, 2);
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 1);
}

#[test]
fn codex_rescan_is_stamp_guarded_and_identity_stable() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    codex_rollout(home.path(), "codex-1", CODEX_BODY, 1_750_000_200_000);
    let first = discover(&conn, home.path(), &only(&["codex"]));
    let second = discover(&conn, home.path(), &only(&["codex"]));
    assert_eq!(first.ids(), second.ids());
    assert_eq!(second.summary.counters.shallow_reads, 0);
    assert_eq!(second.summary.skipped_unchanged, 1);
}

// ---------------------------------------------------------------------------
// cursor
// ---------------------------------------------------------------------------

#[test]
fn cursor_reports_mtime_as_last_activity_and_leaves_first_activity_null() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    cursor_session(
        home.path(),
        "work-app",
        "cursor-1",
        concat!(
            r#"{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\nfix the flaky test\n</user_query>"}]}}"#,
            "\n",
            r#"{"role":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}"#,
            "\n"
        ),
        1_750_000_400_000,
    );

    let found = discover(&conn, home.path(), &only(&["cursor"]));
    let row = found.row("cursor-1");
    assert_eq!(row.first_prompt.as_deref(), Some("fix the flaky test"));
    assert_eq!(row.cwd.as_deref(), Some("/work/app"));
    // Cursor records no per-message timestamps, so mtime is the only signal
    // and it is reported as last activity only.
    assert_eq!(row.last_activity_ms, Some(1_750_000_400_000));
    assert_eq!(row.first_activity_ms, None);
}

#[test]
fn cursor_rescan_keeps_one_row_per_session() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    cursor_session(
        home.path(),
        "work-app",
        "cursor-1",
        "{\"role\":\"user\",\"message\":{\"content\":\"hi\"}}\n",
        1_750_000_400_000,
    );
    discover(&conn, home.path(), &only(&["cursor"]));
    let second = discover(&conn, home.path(), &only(&["cursor"]));
    assert_eq!(second.summary.skipped_unchanged, 1);
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 1);
}

// ---------------------------------------------------------------------------
// grok
// ---------------------------------------------------------------------------

#[test]
fn grok_summary_supplies_identity_and_the_chat_head_supplies_the_prompt() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    grok_session(
        home.path(),
        "%2Fwork%2Fgrok",
        "grok-1",
        r#"{"info":{"id":"grok-1","cwd":"/work/grok"},"created_at":"2026-06-20T09:00:00.000Z","updated_at":"2026-06-20T09:30:00.000Z","head_branch":"trunk"}"#,
        concat!(
            r#"{"type":"user","synthetic_reason":"system_reminder","content":[{"type":"text","text":"synthetic"}]}"#,
            "\n",
            r#"{"type":"user","content":[{"type":"text","text":"summarize the diff"}]}"#,
            "\n",
            r#"{"type":"assistant","content":[{"type":"text","text":"sure"}]}"#,
            "\n"
        ),
        1_750_000_500_000,
    );

    let found = discover(&conn, home.path(), &only(&["grok"]));
    let row = found.row("grok-1");
    assert_eq!(row.cwd.as_deref(), Some("/work/grok"));
    assert_eq!(row.git_branch.as_deref(), Some("trunk"));
    assert_eq!(
        row.first_prompt.as_deref(),
        Some("summarize the diff"),
        "synthetic user turns are not human prompts"
    );
    assert_eq!(
        row.first_activity_ms,
        crate::parse_iso_ms("2026-06-20T09:00:00.000Z")
    );
    assert_eq!(
        row.last_activity_ms,
        crate::parse_iso_ms("2026-06-20T09:30:00.000Z")
    );
}

#[test]
fn grok_rescan_is_stamp_guarded() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    grok_session(
        home.path(),
        "%2Fwork%2Fgrok",
        "grok-1",
        r#"{"info":{"id":"grok-1","cwd":"/work/grok"},"created_at":"2026-06-20T09:00:00.000Z"}"#,
        "{\"type\":\"user\",\"content\":\"hey\"}\n",
        1_750_000_500_000,
    );
    discover(&conn, home.path(), &only(&["grok"]));
    let second = discover(&conn, home.path(), &only(&["grok"]));
    assert_eq!(second.summary.counters.shallow_reads, 0);
    assert_eq!(second.summary.skipped_unchanged, 1);
}

// ---------------------------------------------------------------------------
// opencode
// ---------------------------------------------------------------------------

#[test]
fn opencode_sessions_come_from_the_session_table_with_a_first_prompt() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    opencode_db(
        home.path(),
        r#"INSERT INTO session VALUES ('oc-1', '/work/oc', 1750000600000, 1750000700000);
           INSERT INTO message VALUES ('m1', 'oc-1', 1750000600000, '{"role":"user","modelID":"claude-sonnet"}');
           INSERT INTO part VALUES ('p1', 'm1', 'oc-1', 1750000600000, '{"type":"text","text":"port the parser"}');"#,
    );

    let found = discover(&conn, home.path(), &only(&["opencode"]));
    let row = found.row("oc-1");
    assert_eq!(row.cwd.as_deref(), Some("/work/oc"));
    assert_eq!(row.first_prompt.as_deref(), Some("port the parser"));
    assert_eq!(row.first_activity_ms, Some(1_750_000_600_000));
    assert_eq!(row.last_activity_ms, Some(1_750_000_700_000));
    assert_eq!(row.models, vec!["claude-sonnet".to_string()]);
    assert_eq!(row.raw_path, None, "opencode sessions are not file-backed");

    let second = discover(&conn, home.path(), &only(&["opencode"]));
    assert_eq!(second.summary.skipped_unchanged, 1);
    assert_eq!(second.summary.counters.shallow_reads, 0);
}

// ---------------------------------------------------------------------------
// relay
// ---------------------------------------------------------------------------

#[test]
fn relay_discovers_from_already_synced_rows_and_never_touches_the_network() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    for (id, prompt, ts) in [
        ("ch:general", "[ana] deploy is red", 1_750_000_800_000_i64),
        ("ch:general", "[bo] rolling back", 1_750_000_900_000),
    ] {
        ai_hist_core::insert_history(
            &conn,
            &ai_hist_core::HistoryEntry {
                id: 0,
                source: "relay".into(),
                session_id: Some(id.into()),
                project: Some("ws-1".into()),
                prompt: prompt.into(),
                prompt_hash: None,
                timestamp_ms: ts,
            },
        )
        .unwrap();
    }

    let found = discover(&conn, home.path(), &only(&["relay"]));
    let row = found.row("ch:general");
    assert_eq!(row.first_prompt.as_deref(), Some("[ana] deploy is red"));
    assert_eq!(row.first_activity_ms, Some(1_750_000_800_000));
    assert_eq!(row.last_activity_ms, Some(1_750_000_900_000));
    // Relay has no local working directory to report, and inventing one would
    // be a fabrication.
    assert_eq!(row.cwd, None);
    // No local files were opened at all: relay is a database-only adapter.
    assert_eq!(found.summary.counters.files_opened, 0);
}

#[test]
fn relay_with_nothing_synced_discovers_nothing_rather_than_failing() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    let found = discover(&conn, home.path(), &only(&["relay"]));
    assert!(found.rows.is_empty());
    assert!(found.summary.diagnostics.is_empty());
    assert_eq!(found.summary.providers["relay"].candidates, 0);
}

// ---------------------------------------------------------------------------
// cross-provider
// ---------------------------------------------------------------------------

fn three_providers(home: &Path) {
    claude_session(home, "claude-old", CLAUDE_BODY, 1_000_000_000_000);
    codex_rollout(home, "codex-mid", CODEX_BODY, 2_000_000_000_000);
    cursor_session(
        home,
        "work-app",
        "cursor-new",
        "{\"role\":\"user\",\"message\":{\"content\":\"newest\"}}\n",
        3_000_000_000_000,
    );
}

#[test]
fn candidates_are_ordered_globally_by_recency_across_providers() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    three_providers(home.path());

    let found = discover(&conn, home.path(), &DiscoverOptions::default());
    assert_eq!(
        found.ids(),
        vec![
            "cursor:cursor-new".to_string(),
            "codex:codex-1".to_string(),
            "claude:claude-1".to_string(),
        ],
        "rows must arrive newest-first across providers, not provider by provider"
    );
}

#[test]
fn the_limit_is_global_not_per_provider() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    // Two providers, three sessions each, interleaved in time. The three
    // newest overall are two claude sessions and one codex session.
    for (index, id) in ["claude-a", "claude-b", "claude-c"].iter().enumerate() {
        let body = CLAUDE_BODY.replace("claude-1", id);
        claude_session(
            home.path(),
            id,
            &body,
            1_000_000_000_000 + index as i64 * 100,
        );
    }
    for (index, id) in ["codex-a", "codex-b", "codex-c"].iter().enumerate() {
        let body = CODEX_BODY.replace("codex-1", id);
        codex_rollout(
            home.path(),
            id,
            &body,
            1_000_000_000_050 + index as i64 * 100,
        );
    }

    let found = discover(
        &conn,
        home.path(),
        &DiscoverOptions {
            sources: Vec::new(),
            limit: Some(3),
        },
    );
    assert_eq!(
        found.ids(),
        vec![
            "codex:codex-c".to_string(),
            "claude:claude-c".to_string(),
            "codex:codex-b".to_string(),
        ],
        "the newest three overall (two codex, one claude), not three from one provider"
    );
    assert_eq!(found.summary.counters.candidates_enumerated, 6);
    assert_eq!(found.summary.counters.shallow_reads, 3);
}

#[test]
fn the_same_native_id_under_two_providers_is_two_rows() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(
        home.path(),
        "shared",
        &CLAUDE_BODY.replace("claude-1", "shared"),
        1_000_000_000_000,
    );
    codex_rollout(
        home.path(),
        "shared",
        &CODEX_BODY.replace("codex-1", "shared"),
        1_000_000_000_100,
    );

    let found = discover(&conn, home.path(), &DiscoverOptions::default());
    assert_eq!(found.rows.len(), 2);
    let sources: BTreeSet<&str> = found.rows.iter().map(|row| row.source.as_str()).collect();
    assert_eq!(sources, ["claude", "codex"].into_iter().collect());
    let rows: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE session_id = 'shared'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        rows, 2,
        "(source, session_id) is the identity, not id alone"
    );
}

#[test]
fn one_broken_provider_does_not_block_the_others() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);
    // Not a SQLite database at all: the opencode snapshot must fail.
    fs::write(home.path().join("opencode.db"), "definitely not sqlite").unwrap();

    let found = discover(&conn, home.path(), &DiscoverOptions::default());
    assert_eq!(found.ids(), vec!["claude:claude-1"]);
    assert!(found.summary.providers["opencode"].failed);
    assert!(!found.summary.providers["claude"].failed);
    assert_eq!(found.summary.diagnostics.len(), 1);
    assert_eq!(found.summary.diagnostics[0].source, "opencode");
}

#[test]
fn a_run_fails_only_when_every_provider_fails() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    fs::write(home.path().join("opencode.db"), "definitely not sqlite").unwrap();
    let env = env_at(&conn, home.path());
    let error = discover_sessions_with_env(&env, &only(&["opencode"]), |_| {}).unwrap_err();
    assert!(
        format!("{error:#}").contains("no provider made progress"),
        "{error:#}"
    );
}

#[test]
fn bounded_reads_do_not_grow_with_the_size_of_the_archive() {
    let conn = catalog();
    let small = tempfile::tempdir().unwrap();
    claude_session(small.path(), "a", &CLAUDE_BODY.replace("claude-1", "a"), 10);
    codex_rollout(small.path(), "b", &CODEX_BODY.replace("codex-1", "b"), 20);
    let limited = DiscoverOptions {
        sources: Vec::new(),
        limit: Some(2),
    };
    let baseline = discover(&conn, small.path(), &limited);

    let conn = catalog();
    let big = tempfile::tempdir().unwrap();
    for index in 0..50 {
        let id = format!("bulk-{index:02}");
        claude_session(
            big.path(),
            &id,
            &CLAUDE_BODY.replace("claude-1", &id),
            1_000 + index as i64,
        );
    }
    claude_session(
        big.path(),
        "a",
        &CLAUDE_BODY.replace("claude-1", "a"),
        900_000,
    );
    codex_rollout(
        big.path(),
        "b",
        &CODEX_BODY.replace("codex-1", "b"),
        900_010,
    );
    let scaled = discover(&conn, big.path(), &limited);

    assert_eq!(scaled.summary.counters.candidates_enumerated, 52);
    assert_eq!(
        scaled.summary.counters.shallow_reads, 2,
        "a limit of 2 must read exactly two sources, whatever the archive holds"
    );
    assert_eq!(
        scaled.summary.counters.files_opened, baseline.summary.counters.files_opened,
        "file opens must not scale with the archive"
    );
    assert_eq!(
        scaled.summary.counters.bytes_read, baseline.summary.counters.bytes_read,
        "bytes read must not scale with the archive"
    );
    assert_eq!(
        scaled.ids(),
        vec!["codex:b".to_string(), "claude:a".to_string()]
    );
}

#[test]
fn a_head_read_stays_inside_its_budget_on_a_very_large_transcript() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    let mut body = String::from(CLAUDE_BODY);
    let filler = r#"{"sessionId":"claude-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"PADDINGPADDINGPADDINGPADDINGPADDINGPADDINGPADDINGPADDING"}]},"timestamp":"2026-06-20T10:06:00.000Z"}"#;
    while body.len() < 3 * HEAD_SCAN_MAX_BYTES as usize {
        body.push_str(filler);
        body.push('\n');
    }
    body.push_str(
        r#"{"sessionId":"claude-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"final"}]},"timestamp":"2026-06-20T23:00:00.000Z"}"#,
    );
    body.push('\n');
    let path = claude_session(home.path(), "claude-1", &body, 1_750_000_000_000);
    let size = fs::metadata(&path).unwrap().len();

    let found = discover(&conn, home.path(), &only(&["claude"]));
    assert!(
        found.summary.counters.bytes_read <= HEAD_SCAN_MAX_BYTES + TAIL_SCAN_MAX_BYTES,
        "read {} bytes of a {size}-byte transcript",
        found.summary.counters.bytes_read
    );
    assert!(found.summary.counters.bytes_read < size);
    let row = found.row("claude-1");
    assert_eq!(row.first_prompt.as_deref(), Some("the real first prompt"));
    assert_eq!(
        row.last_activity_ms,
        crate::parse_iso_ms("2026-06-20T23:00:00.000Z"),
        "the tail read must still find the last timestamp"
    );
}

// ---------------------------------------------------------------------------
// catalog listing
// ---------------------------------------------------------------------------

#[test]
fn the_cache_only_listing_survives_the_provider_files_disappearing() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    three_providers(home.path());
    discover(&conn, home.path(), &DiscoverOptions::default());

    // Nothing on disk any more: a cache-only list must not care.
    fs::remove_dir_all(home.path().join(".claude")).unwrap();
    fs::remove_dir_all(home.path().join(".codex")).unwrap();
    fs::remove_dir_all(home.path().join(".cursor")).unwrap();

    let rows = list_session_catalog(&conn, &CatalogListOptions::default()).unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.from_cache));
    let recency: Vec<Option<i64>> = rows.iter().map(|row| row.last_activity_ms).collect();
    let mut sorted = recency.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(recency, sorted, "the catalog lists newest first");
}

#[test]
fn the_catalog_query_reads_only_the_sessions_table() {
    let sql = catalog_list_sql(&CatalogListOptions::default());
    assert!(sql.contains("FROM sessions"), "{sql}");
    for forbidden in ["history", "session_events", "tool_calls", "file_edits"] {
        assert!(
            !sql.contains(forbidden),
            "the cache-only listing must not touch {forbidden}: {sql}"
        );
    }
}

#[test]
fn trajectory_rows_never_appear_in_a_session_listing() {
    let conn = catalog();
    conn.execute(
        "INSERT INTO sessions (session_id, source, last_activity_ms, discovery_state) \
         VALUES ('traj-1', 'trajectory', 9999999999999, 'full')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, last_activity_ms, discovery_state) \
         VALUES ('claude-1', 'claude', 1, 'shallow')",
        [],
    )
    .unwrap();
    let rows = list_session_catalog(&conn, &CatalogListOptions::default()).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].source, "claude");
}

#[test]
fn the_catalog_listing_filters_by_source_and_paginates_by_recency() {
    let conn = catalog();
    for (source, id, ts) in [
        ("claude", "c1", 300_i64),
        ("claude", "c2", 200),
        ("codex", "x1", 250),
    ] {
        conn.execute(
            "INSERT INTO sessions (session_id, source, last_activity_ms, discovery_state) \
             VALUES (?, ?, ?, 'shallow')",
            params![id, source, ts],
        )
        .unwrap();
    }
    let claude_only = list_session_catalog(
        &conn,
        &CatalogListOptions {
            sources: vec!["claude".into()],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        claude_only
            .iter()
            .map(|row| row.session_id.clone())
            .collect::<Vec<_>>(),
        vec!["c1", "c2"]
    );

    let page = list_session_catalog(
        &conn,
        &CatalogListOptions {
            limit: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(page[0].session_id, "c1");
    let next = list_session_catalog(
        &conn,
        &CatalogListOptions {
            limit: Some(1),
            before_ms: page[0].last_activity_ms,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(next[0].session_id, "x1");
}

#[test]
fn the_catalog_listing_is_served_by_an_index_not_a_table_scan() {
    let conn = catalog();
    let plan = |options: &CatalogListOptions, args: Vec<Box<dyn rusqlite::ToSql>>| -> String {
        let sql = format!("EXPLAIN QUERY PLAN {}", catalog_list_sql(options));
        let mut stmt = conn.prepare(&sql).unwrap();
        let params = rusqlite::params_from_iter(args.iter().map(|arg| arg.as_ref()));
        stmt.query_map(params, |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
            .join(" | ")
    };

    let unfiltered = plan(&CatalogListOptions::default(), vec![Box::new(10_i64)]);
    assert!(
        unfiltered.contains("idx_sessions_last"),
        "the recency ordering must be served by idx_sessions_last: {unfiltered}"
    );
    assert!(
        !unfiltered.contains("TEMP B-TREE"),
        "the listing must not sort the table: {unfiltered}"
    );

    let filtered = plan(
        &CatalogListOptions {
            sources: vec!["claude".into()],
            limit: Some(10),
            before_ms: Some(1),
        },
        vec![
            Box::new("claude".to_string()),
            Box::new(1_i64),
            Box::new(10_i64),
        ],
    );
    assert!(
        filtered.contains("SEARCH") && filtered.contains("idx_sessions_source_last"),
        "a source-filtered listing must search idx_sessions_source_last: {filtered}"
    );
}

// ---------------------------------------------------------------------------
// full-ingest interaction
// ---------------------------------------------------------------------------

#[test]
fn a_shallow_rescan_never_downgrades_a_fully_indexed_row() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    let path = claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);
    // Stand in for the full-sync path having already ingested this session.
    crate::upsert_session(
        &conn,
        "claude-1",
        "claude",
        Some("/work/app"),
        Some("main"),
        1,
        2,
        Some("the assistant's last word"),
        Some(&path.to_string_lossy()),
    )
    .unwrap();

    let found = discover(&conn, home.path(), &only(&["claude"]));
    let row = found.row("claude-1");
    assert_eq!(
        row.discovery_state, "full",
        "shallow discovery must never claim a fully indexed session is shallow"
    );
    assert_eq!(
        row.last_assistant_text.as_deref(),
        Some("the assistant's last word"),
        "a shallow pass must not null out what full indexing established"
    );
    assert_eq!(
        row.first_prompt.as_deref(),
        Some("the real first prompt"),
        "a shallow rescan of a full row still refreshes catalog metadata"
    );
    assert!(
        row.source_stamp
            .as_deref()
            .is_some_and(|stamp| stamp.starts_with(&format!("v{SHALLOW_SCANNER_VERSION}:"))),
        "the stamp records which scanner wrote it: {:?}",
        row.source_stamp
    );
}

#[test]
fn bumping_the_scanner_version_invalidates_stored_stamps() {
    let conn = catalog();
    let home = tempfile::tempdir().unwrap();
    claude_session(home.path(), "claude-1", CLAUDE_BODY, 1_750_000_000_000);
    discover(&conn, home.path(), &only(&["claude"]));
    // Simulate a database stamped by an older scanner generation.
    conn.execute(
        "UPDATE sessions SET source_stamp = 'v0:stale' WHERE session_id = 'claude-1'",
        [],
    )
    .unwrap();
    let rescan = discover(&conn, home.path(), &only(&["claude"]));
    assert_eq!(rescan.summary.counters.shallow_reads, 1);
    assert_eq!(rescan.summary.skipped_unchanged, 0);
}
