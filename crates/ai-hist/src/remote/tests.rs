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
        Box::new(transport),
        None,
    );
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let error = provider.enumerate(&env).unwrap_err().to_string();
    assert!(error.contains("expired"), "{error}");
}

#[test]
fn claude_transport_refuses_plaintext_off_loopback() {
    assert!(require_https_or_loopback("https://api.anthropic.com").is_ok());
    assert!(require_https_or_loopback("http://127.0.0.1:8787").is_ok());
    assert!(require_https_or_loopback("http://localhost:1234").is_ok());
    let error = require_https_or_loopback("http://api.evil.test").unwrap_err();
    assert!(error.to_string().contains("plain http"));
    // Userinfo must not smuggle a loopback-looking authority past the check.
    assert!(require_https_or_loopback("http://127.0.0.1@evil.test").is_err());
}

// ---------------------------------------------------------------------------
// codex-cloud
// ---------------------------------------------------------------------------

struct ScriptedLister {
    payload: String,
    limits: Mutex<Vec<Option<usize>>>,
}

impl CodexCloudLister for Arc<ScriptedLister> {
    fn list_json(&self, limit: Option<usize>) -> Result<String> {
        self.limits.lock().unwrap().push(limit);
        Ok(self.payload.clone())
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
    assert_eq!(parse_codex_cloud_listing(CODEX_LISTING).unwrap().len(), 2);
    assert_eq!(
        parse_codex_cloud_listing(r#"[{"id":"task_e_1"}]"#)
            .unwrap()
            .len(),
        1
    );
    assert!(parse_codex_cloud_listing("not json").is_err());
    assert!(parse_codex_cloud_listing(r#"{"cursor":null}"#).is_err());
}

#[test]
fn codex_mapping_carries_the_task_listing_and_stamps_on_status() {
    let tasks = parse_codex_cloud_listing(CODEX_LISTING).unwrap();
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
}

#[test]
fn codex_provider_forwards_the_row_limit_to_the_cli() {
    let lister = Arc::new(ScriptedLister {
        payload: CODEX_LISTING.to_string(),
        limits: Mutex::new(Vec::new()),
    });
    let provider = CodexCloudProvider::new(Box::new(Arc::clone(&lister)), Some(7));
    let home = tempfile::tempdir().unwrap();
    let conn = catalog();
    let env = env_at(&conn, home.path());
    let candidates = provider.enumerate(&env).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(*lister.limits.lock().unwrap(), vec![Some(7)]);
}

// ---------------------------------------------------------------------------
// availability
// ---------------------------------------------------------------------------

#[test]
fn statuses_report_missing_credentials_with_the_paths_looked_at() {
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

// ---------------------------------------------------------------------------
// engine integration
// ---------------------------------------------------------------------------

fn remote_codex_provider(payload: &str, limit: Option<usize>) -> Box<dyn ShallowSessionProvider> {
    Box::new(CodexCloudProvider::new(
        Box::new(Arc::new(ScriptedLister {
            payload: payload.to_string(),
            limits: Mutex::new(Vec::new()),
        })),
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
