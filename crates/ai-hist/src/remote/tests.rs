//! Tests for the remote session connectors.
//!
//! The network and process boundaries are faked ([`ClaudeSessionsTransport`],
//! [`CodexCloudLister`]), so mapping, pagination, and the engine integration
//! are asserted without a claude.ai account or the Codex CLI installed. The
//! real transports are exercised end-to-end by `tests/session_discovery.rs`
//! (a scripted `codex` binary and a loopback HTTP server).

use super::*;
use crate::discover::{
    discover_sessions_with_providers, list_session_catalog, CatalogListOptions, DiscoverOptions,
};
use ai_hist_core::{init_db, SessionScope};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn catalog() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory database");
    init_db(&conn).expect("schema");
    conn
}

fn env_at<'a>(conn: &'a Connection, home: &Path) -> DiscoveryEnv<'a> {
    DiscoveryEnv::with_roots(conn, home.to_path_buf(), home.join("opencode.db"))
}

fn write_claude_credentials(home: &Path, expires_at_ms: i64) -> PathBuf {
    let path = home.join(".claude/.credentials.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        format!(
            r#"{{"claudeAiOauth":{{"accessToken":"sk-ant-oat01-test","refreshToken":"sk-ant-ort01-test","expiresAt":{expires_at_ms},"scopes":["user:inference"]}}}}"#
        ),
    )
    .unwrap();
    path
}

const FAR_FUTURE_MS: i64 = 4_102_444_800_000; // 2100-01-01

/// Serializes the availability tests and clears the ambient
/// `RELAYHISTORY_CLAUDE_CREDENTIALS` override for their duration, restoring
/// whatever value the process had once the guard drops.
static CREDENTIALS_OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

struct ClearedCredentialsOverride {
    previous: Option<std::ffi::OsString>,
    _serialized: std::sync::MutexGuard<'static, ()>,
}

fn without_credentials_override() -> ClearedCredentialsOverride {
    let guard = CREDENTIALS_OVERRIDE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous = std::env::var_os("RELAYHISTORY_CLAUDE_CREDENTIALS");
    std::env::remove_var("RELAYHISTORY_CLAUDE_CREDENTIALS");
    ClearedCredentialsOverride {
        previous,
        _serialized: guard,
    }
}

impl Drop for ClearedCredentialsOverride {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("RELAYHISTORY_CLAUDE_CREDENTIALS", value),
            None => std::env::remove_var("RELAYHISTORY_CLAUDE_CREDENTIALS"),
        }
    }
}

// ---------------------------------------------------------------------------
// claude-web
// ---------------------------------------------------------------------------

/// Serves a fixed sequence of responses and records each requested URL.
struct ScriptedTransport {
    responses: Vec<(u16, String)>,
    calls: Mutex<Vec<(String, String)>>,
    next: AtomicUsize,
}

impl ScriptedTransport {
    fn new(responses: Vec<(u16, String)>) -> Self {
        Self {
            responses,
            calls: Mutex::new(Vec::new()),
            next: AtomicUsize::new(0),
        }
    }
}

impl ClaudeSessionsTransport for Arc<ScriptedTransport> {
    fn get(&self, url: &str, bearer_token: &str) -> Result<ClaudeHttpResponse> {
        self.calls
            .lock()
            .unwrap()
            .push((url.to_string(), bearer_token.to_string()));
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        let (status, body) = self
            .responses
            .get(index)
            .cloned()
            .unwrap_or((500, "script exhausted".to_string()));
        Ok(ClaudeHttpResponse { status, body })
    }
}

fn claude_page(sessions: &[Value], next_cursor: Option<&str>) -> String {
    serde_json::json!({
        "data": sessions,
        "next_cursor": next_cursor,
    })
    .to_string()
}

fn web_session(id: &str, title: &str, last_event_at: &str) -> Value {
    serde_json::json!({
        "id": id,
        "title": title,
        "status": "idle",
        "worker_status": "idle",
        "created_at": "2026-06-20T09:00:00Z",
        "last_event_at": last_event_at,
        "environment_kind": "cloud",
        "config": {
            "sources": [
                {"type": "git_repository", "url": "https://github.com/acme/api"}
            ]
        }
    })
}

fn claude_provider(
    home: &Path,
    transport: &Arc<ScriptedTransport>,
    limit: Option<usize>,
) -> ClaudeWebProvider {
    ClaudeWebProvider::new(
        write_claude_credentials(home, FAR_FUTURE_MS),
        "https://api.example.test".to_string(),
        Box::new(Arc::clone(transport)),
        limit,
    )
}

#[test]
fn claude_mapping_carries_observed_fields_and_nothing_invented() {
    let (candidate, session) = map_claude_web_session(&web_session(
        "session_01abc",
        "Fix login flow",
        "2026-06-21T10:00:00Z",
    ))
    .expect("a mappable session");
    assert_eq!(session.source, "claude");
    assert_eq!(session.session_id, "session_01abc");
    assert_eq!(session.first_prompt.as_deref(), Some("Fix login flow"));
    assert_eq!(
        session.repo_url.as_deref(),
        Some("https://github.com/acme/api")
    );
    assert_eq!(
        session.raw_path.as_deref(),
        Some("https://claude.ai/code/session_01abc")
    );
    assert_eq!(session.discovery_state, "shallow");
    assert_eq!(session.cwd, None);
    assert_eq!(session.git_branch, None);
    assert!(session.models.is_empty());
    assert!(session.first_activity_ms.unwrap() < session.last_activity_ms.unwrap());
    assert_eq!(candidate.session_id.as_deref(), Some("session_01abc"));
    assert_eq!(candidate.recency_hint_ms, session.last_activity_ms);
    assert_eq!(candidate.stamp, "web:2026-06-21T10:00:00Z");
}

#[test]
fn claude_mapping_skips_bridges_and_malformed_ids() {
    let mut bridge = web_session("session_01abc", "t", "2026-06-21T10:00:00Z");
    bridge["environment_kind"] = "bridge".into();
    assert!(map_claude_web_session(&bridge).is_none());

    for bad_id in ["", "sess_x", "session_", "session_a b", "task_e_1"] {
        let entry = web_session(bad_id, "t", "2026-06-21T10:00:00Z");
        assert!(
            map_claude_web_session(&entry).is_none(),
            "id {bad_id:?} must not map"
        );
    }
    assert!(
        map_claude_web_session(&web_session("cse_9X-y_z", "t", "2026-06-21T10:00:00Z")).is_some()
    );
}

#[test]
fn claude_enumeration_pages_until_the_cursor_ends() {
    let home = tempfile::tempdir().unwrap();
    let transport = Arc::new(ScriptedTransport::new(vec![
        (
            200,
            claude_page(
                &[web_session("session_01", "one", "2026-06-21T10:00:00Z")],
                Some("cursor with spaces"),
            ),
        ),
        (
            200,
            claude_page(
                &[web_session("session_02", "two", "2026-06-20T10:00:00Z")],
                None,
            ),
        ),
    ]));
    let provider = claude_provider(home.path(), &transport, None);
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let candidates = provider.enumerate(&env).unwrap();
    assert_eq!(candidates.len(), 2);
    let calls = transport.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].0.ends_with("/v1/code/sessions?limit=100"));
    assert!(calls[1].0.contains("cursor=cursor%20with%20spaces"));
    assert!(calls.iter().all(|(_, token)| token == "sk-ant-oat01-test"));
}

#[test]
fn claude_enumeration_respects_the_row_limit() {
    let home = tempfile::tempdir().unwrap();
    let transport = Arc::new(ScriptedTransport::new(vec![(
        200,
        claude_page(
            &[
                web_session("session_01", "one", "2026-06-21T10:00:00Z"),
                web_session("session_02", "two", "2026-06-20T10:00:00Z"),
            ],
            Some("more"),
        ),
    )]));
    let provider = claude_provider(home.path(), &transport, Some(2));
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let candidates = provider.enumerate(&env).unwrap();
    // The limit is satisfied by the first page, so the cursor is not followed.
    assert_eq!(candidates.len(), 2);
    let calls = transport.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].0.ends_with("?limit=2"));
}

#[test]
fn claude_enumeration_reports_a_rejected_token() {
    let home = tempfile::tempdir().unwrap();
    let transport = Arc::new(ScriptedTransport::new(vec![(401, "{}".to_string())]));
    let provider = claude_provider(home.path(), &transport, None);
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let error = provider.enumerate(&env).unwrap_err().to_string();
    assert!(error.contains("rejected the stored OAuth token"), "{error}");
}

#[test]
fn claude_enumeration_reports_an_expired_token_without_a_request() {
    let home = tempfile::tempdir().unwrap();
    let transport = Arc::new(ScriptedTransport::new(vec![]));
    let provider = ClaudeWebProvider::new(
        write_claude_credentials(home.path(), 1_000),
        "https://api.example.test".to_string(),
        Box::new(Arc::clone(&transport)),
        None,
    );
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let error = provider.enumerate(&env).unwrap_err().to_string();
    assert!(error.contains("expired"), "{error}");
    assert!(
        transport.calls.lock().unwrap().is_empty(),
        "an expired token must be rejected before any request is made"
    );
}

#[test]
fn claude_transport_refuses_plaintext_off_loopback() {
    assert!(require_https_or_loopback("https://api.anthropic.com").is_ok());
    assert!(require_https_or_loopback("http://127.0.0.1:8787").is_ok());
    assert!(require_https_or_loopback("http://localhost:1234").is_ok());
    assert!(require_https_or_loopback("http://[::1]:8787").is_ok());
    let error = require_https_or_loopback("http://api.evil.test").unwrap_err();
    assert!(error.to_string().contains("plain http"));
    // Userinfo must not smuggle a loopback-looking authority past the check.
    assert!(require_https_or_loopback("http://127.0.0.1@evil.test").is_err());
}

// ---------------------------------------------------------------------------
// codex-cloud
// ---------------------------------------------------------------------------

/// Serves a fixed sequence of listing pages and records each (limit, cursor)
/// the provider asked for.
struct ScriptedLister {
    pages: Vec<String>,
    calls: Mutex<Vec<(usize, Option<String>)>>,
    next: AtomicUsize,
}

impl ScriptedLister {
    fn new(pages: Vec<String>) -> Self {
        Self {
            pages,
            calls: Mutex::new(Vec::new()),
            next: AtomicUsize::new(0),
        }
    }
}

impl CodexCloudLister for Arc<ScriptedLister> {
    fn list_json(&self, limit: usize, cursor: Option<&str>) -> Result<String> {
        self.calls
            .lock()
            .unwrap()
            .push((limit, cursor.map(str::to_string)));
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .pages
            .get(index)
            .cloned()
            .unwrap_or_else(|| r#"{"tasks":[],"cursor":null}"#.to_string()))
    }
}

const CODEX_LISTING: &str = r#"{
  "tasks": [
    {
      "id": "task_e_123",
      "url": "https://chatgpt.com/codex/tasks/task_e_123",
      "title": "Fix the flaky retry test",
      "status": "ready",
      "updated_at": "2026-06-22T09:00:00Z",
      "environment_id": "env_1",
      "environment_label": "api",
      "summary": "1 file changed",
      "is_review": false,
      "attempt_total": 1
    },
    {"title": "no id, not a task"}
  ],
  "cursor": null
}"#;

#[test]
fn codex_listing_parses_both_documented_shapes() {
    let page = parse_codex_cloud_listing(CODEX_LISTING).unwrap();
    assert_eq!(page.tasks.len(), 2);
    assert_eq!(page.cursor, None);
    let bare = parse_codex_cloud_listing(r#"[{"id":"task_e_1"}]"#).unwrap();
    assert_eq!(bare.tasks.len(), 1);
    assert_eq!(bare.cursor, None, "a bare array carries no continuation");
    let continued =
        parse_codex_cloud_listing(r#"{"tasks":[{"id":"task_e_2"}],"cursor":"page-2"}"#).unwrap();
    assert_eq!(continued.cursor.as_deref(), Some("page-2"));
    assert!(parse_codex_cloud_listing("not json").is_err());
    assert!(parse_codex_cloud_listing(r#"{"cursor":null}"#).is_err());
}

#[test]
fn codex_mapping_carries_the_task_listing_and_stamps_on_status() {
    let tasks = parse_codex_cloud_listing(CODEX_LISTING).unwrap().tasks;
    let (candidate, session) = map_codex_cloud_task(&tasks[0]).expect("a mappable task");
    assert_eq!(session.source, "codex");
    assert_eq!(session.session_id, "task_e_123");
    assert_eq!(
        session.first_prompt.as_deref(),
        Some("Fix the flaky retry test")
    );
    assert_eq!(
        session.raw_path.as_deref(),
        Some("https://chatgpt.com/codex/tasks/task_e_123")
    );
    assert_eq!(session.cwd, None);
    assert_eq!(candidate.stamp, "cloud:2026-06-22T09:00:00Z:ready");
    // An id-less entry is not a task.
    assert!(map_codex_cloud_task(&tasks[1]).is_none());

    // A pathological title is bounded like every stored excerpt.
    let long = serde_json::json!({"id": "task_e_long", "title": "x".repeat(9000)});
    let (_, session) = map_codex_cloud_task(&long).unwrap();
    assert_eq!(
        session.first_prompt.unwrap().chars().count(),
        crate::discover::EXCERPT_MAX_CHARS
    );
    let long_web = {
        let mut entry = web_session("session_01long", "t", "2026-06-21T10:00:00Z");
        entry["title"] = serde_json::Value::String("y".repeat(9000));
        entry
    };
    let (_, session) = map_claude_web_session(&long_web).unwrap();
    assert_eq!(
        session.first_prompt.unwrap().chars().count(),
        crate::discover::EXCERPT_MAX_CHARS
    );
}

#[test]
fn codex_provider_forwards_a_bounded_page_limit_to_the_cli() {
    let lister = Arc::new(ScriptedLister::new(vec![CODEX_LISTING.to_string()]));
    let provider = CodexCloudProvider::new(Box::new(Arc::clone(&lister)), Some(7));
    let home = tempfile::tempdir().unwrap();
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let candidates = provider.enumerate(&env).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(*lister.calls.lock().unwrap(), vec![(7, None)]);

    // A global limit above the CLI's window is clamped per page, never
    // forwarded verbatim (the CLI rejects --limit values over 20).
    let lister = Arc::new(ScriptedLister::new(vec![CODEX_LISTING.to_string()]));
    let provider = CodexCloudProvider::new(Box::new(Arc::clone(&lister)), Some(500));
    let env = env_at(&conn, home.path());
    provider.enumerate(&env).unwrap();
    assert_eq!(*lister.calls.lock().unwrap(), vec![(20, None)]);
}

#[test]
fn codex_provider_follows_the_cursor_across_pages() {
    let page_one = r#"{"tasks":[{"id":"task_e_1","title":"one","status":"ready","updated_at":"2026-06-22T09:00:00Z"}],"cursor":"page-2"}"#;
    let page_two = r#"{"tasks":[{"id":"task_e_2","title":"two","status":"ready","updated_at":"2026-06-21T09:00:00Z"}],"cursor":null}"#;
    let lister = Arc::new(ScriptedLister::new(vec![
        page_one.to_string(),
        page_two.to_string(),
    ]));
    let provider = CodexCloudProvider::new(Box::new(Arc::clone(&lister)), None);
    let home = tempfile::tempdir().unwrap();
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let candidates = provider.enumerate(&env).unwrap();
    assert_eq!(candidates.len(), 2);
    assert_eq!(
        *lister.calls.lock().unwrap(),
        vec![(20, None), (20, Some("page-2".to_string()))]
    );

    // A satisfied row limit ends the walk without following the cursor.
    let lister = Arc::new(ScriptedLister::new(vec![
        page_one.to_string(),
        page_two.to_string(),
    ]));
    let provider = CodexCloudProvider::new(Box::new(Arc::clone(&lister)), Some(1));
    let env = env_at(&conn, home.path());
    let candidates = provider.enumerate(&env).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(*lister.calls.lock().unwrap(), vec![(1, None)]);
}

// ---------------------------------------------------------------------------
// availability
// ---------------------------------------------------------------------------

#[test]
fn statuses_report_missing_credentials_with_the_paths_looked_at() {
    // A developer's ambient credentials override must not leak into the
    // isolated home this test asserts against.
    let _cleared = without_credentials_override();
    let home = tempfile::tempdir().unwrap();
    let statuses = remote_connector_statuses_at(home.path());
    assert_eq!(statuses.len(), 2);
    assert!(statuses.iter().all(|status| !status.configured));
    let error = ensure_remote_connectors_configured_at("discovery", home.path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("no remote provider connectors are configured"),
        "{error}"
    );
    assert!(error.contains("claude-web"), "{error}");
    assert!(error.contains("codex-cloud"), "{error}");

    std::fs::create_dir_all(home.path().join(".codex")).unwrap();
    std::fs::write(home.path().join(".codex/auth.json"), "{}").unwrap();
    let statuses = remote_connector_statuses_at(home.path());
    assert!(statuses.iter().any(|status| status.configured));
    assert!(ensure_remote_connectors_configured_at("discovery", home.path()).is_ok());
}

#[test]
fn a_source_filter_that_excludes_every_configured_connector_is_unsupported() {
    let _cleared = without_credentials_override();
    let home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(home.path().join(".codex")).unwrap();
    std::fs::write(home.path().join(".codex/auth.json"), "{}").unwrap();

    let ok = |sources: &[&str]| {
        ensure_remote_connectors_configured_for_at(
            "discovery",
            home.path(),
            &sources.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
    };
    assert!(ok(&[]).is_ok());
    assert!(ok(&["codex"]).is_ok());
    assert!(
        ok(&["codex", "claude"]).is_ok(),
        "one configured match is enough"
    );

    // Only codex is configured, so a claude-only request is unsupported…
    let error = ok(&["claude"]).unwrap_err().to_string();
    assert!(
        error.contains("no remote provider connectors are configured"),
        "{error}"
    );
    assert!(error.contains("claude-web"), "{error}");
    assert!(!error.contains("codex-cloud"), "{error}");
    // …and a source no connector will ever serve says so distinctly.
    let error = ok(&["cursor"]).unwrap_err().to_string();
    assert!(
        error.contains("no matching remote provider connectors exist"),
        "{error}"
    );
    // A misspelled source is an invalid argument, not an unsupported request.
    let error = ok(&["bogus"]).unwrap_err().to_string();
    assert!(error.contains("invalid source 'bogus'"), "{error}");
}

// ---------------------------------------------------------------------------
// engine integration
// ---------------------------------------------------------------------------

fn remote_codex_provider(payload: &str, limit: Option<usize>) -> Box<dyn ShallowSessionProvider> {
    Box::new(CodexCloudProvider::new(
        Box::new(Arc::new(ScriptedLister::new(vec![payload.to_string()]))),
        limit,
    ))
}

#[test]
fn remote_rows_land_with_a_remote_presence_and_skip_on_an_unchanged_stamp() {
    let home = tempfile::tempdir().unwrap();
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let options = DiscoverOptions {
        scope: SessionScope::Remote,
        ..Default::default()
    };

    let providers = vec![remote_codex_provider(CODEX_LISTING, None)];
    let mut rows = Vec::new();
    let summary = discover_sessions_with_providers(&env, &options, &providers, |session| {
        rows.push(session.clone())
    })
    .unwrap();
    assert_eq!(summary.locations_run, ["remote"]);
    assert_eq!(summary.discovered, 1);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].locations, ["remote"]);
    assert_eq!(rows[0].discovery_state, "shallow");
    assert_eq!(summary.counters.files_opened, 0);

    // Same listing again: the stamp matches the stored remote presence, so the
    // row is served from the catalog without a fresh "read".
    let providers = vec![remote_codex_provider(CODEX_LISTING, None)];
    let env = env_at(&conn, home.path());
    let summary = discover_sessions_with_providers(&env, &options, &providers, |_| {}).unwrap();
    assert_eq!(summary.discovered, 0);
    assert_eq!(summary.skipped_unchanged, 1);

    // A status change alone re-reads the task even though the timestamp is
    // unchanged.
    let changed = CODEX_LISTING.replace("\"ready\"", "\"applied\"");
    let providers = vec![remote_codex_provider(&changed, None)];
    let env = env_at(&conn, home.path());
    let summary = discover_sessions_with_providers(&env, &options, &providers, |_| {}).unwrap();
    assert_eq!(summary.discovered, 1);
}

/// A minimal local adapter for engine tests: one prebuilt session, emitted
/// as a candidate with its id known up front.
struct FakeLocalProvider {
    session: ShallowSession,
    stamp: &'static str,
}

impl ShallowSessionProvider for FakeLocalProvider {
    fn source(&self) -> &'static str {
        "codex"
    }

    fn enumerate(&self, _env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        Ok(vec![Candidate {
            source: "codex",
            locator: self.session.raw_path.clone().unwrap_or_default(),
            session_id: Some(self.session.session_id.clone()),
            // Newer than the remote candidate, so the local read lands first
            // in the window and the remote upsert is the later merge.
            recency_hint_ms: Some(2_000_000_000_000),
            stamp: self.stamp.to_string(),
        }])
    }

    fn read_shallow(
        &self,
        _scan: &ScanEnv<'_>,
        _catalog: Option<&Connection>,
        _candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        Ok(Some(self.session.clone()))
    }
}

#[test]
fn one_window_merging_local_and_remote_emits_the_fully_merged_row() {
    let home = tempfile::tempdir().unwrap();
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let providers: Vec<Box<dyn ShallowSessionProvider>> = vec![
        Box::new(FakeLocalProvider {
            session: ShallowSession {
                source: "codex".into(),
                session_id: "task_e_123".into(),
                cwd: Some("/work/api".into()),
                raw_path: Some("/home/x/.codex/sessions/rollout.jsonl".into()),
                ..Default::default()
            },
            stamp: "local-stamp",
        }),
        remote_codex_provider(CODEX_LISTING, None),
    ];
    let options = DiscoverOptions {
        scope: SessionScope::All,
        ..Default::default()
    };
    let mut rows = Vec::new();
    let summary = discover_sessions_with_providers(&env, &options, &providers, |session| {
        rows.push(session.clone())
    })
    .unwrap();
    assert_eq!(summary.locations_run, ["local", "remote"]);
    // One logical session reached through both adapters in one window: the
    // emitted row must be the final merged state, not the first upsert.
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert_eq!(rows[0].locations, ["local", "remote"]);
    assert_eq!(rows[0].cwd.as_deref(), Some("/work/api"));
    assert_eq!(
        summary.discovered, 2,
        "both adapters performed a fresh read"
    );
}

#[test]
fn a_session_seen_locally_and_remotely_is_one_row_with_both_presences() {
    let home = tempfile::tempdir().unwrap();
    let conn = catalog();

    // The same codex session id observed locally first…
    let local = ShallowSession {
        source: "codex".into(),
        session_id: "task_e_123".into(),
        cwd: Some("/work/api".into()),
        raw_path: Some("/home/x/.codex/sessions/rollout.jsonl".into()),
        source_stamp: Some("v2:local".into()),
        discovery_state: "shallow".into(),
        ..Default::default()
    };
    crate::discover::upsert_shallow_session(&conn, &local).unwrap();

    // …then discovered remotely.
    let env = env_at(&conn, home.path());
    let options = DiscoverOptions {
        scope: SessionScope::Remote,
        ..Default::default()
    };
    let providers = vec![remote_codex_provider(CODEX_LISTING, None)];
    let mut rows = Vec::new();
    discover_sessions_with_providers(&env, &options, &providers, |session| {
        rows.push(session.clone())
    })
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].locations, ["local", "remote"]);
    // The remote pass must not clobber locally observed metadata.
    assert_eq!(rows[0].cwd.as_deref(), Some("/work/api"));

    // Scoped listings serve the one canonical row from either side.
    let remote_only = list_session_catalog(
        &conn,
        &CatalogListOptions {
            scope: SessionScope::Remote,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(remote_only.len(), 1);
    let local_only = list_session_catalog(
        &conn,
        &CatalogListOptions {
            scope: SessionScope::Local,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(local_only.len(), 1);
}
