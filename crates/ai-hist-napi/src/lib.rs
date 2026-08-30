//! Mandatory Node-API boundary for RelayHistory.
//!
//! This crate maps typed JavaScript requests to reusable Rust operations. It
//! contains no SQL, provider parsing, migration, or query semantics of its own.
#![deny(clippy::all)]

use std::path::{Path, PathBuf};

use ai_hist_core::{
    default_db_path, open_db, open_db_readonly, recent as core_recent, schema_is_current,
    search as core_search, session as core_session,
    session_events_page as core_session_events_page, stats as core_stats, HistoryEntry,
    QueryFilter, SessionEvent as CoreSessionEvent, SessionEventCursor as CoreEventCursor,
};
use napi_derive::napi;

/// Bump whenever native object shapes or semantics require an SDK change.
pub const NATIVE_CONTRACT_VERSION: u32 = 2;
const DEFAULT_LIMIT: i64 = 50;
const DEFAULT_EVENT_LIMIT: i64 = 200;

fn db_path(path: Option<String>) -> PathBuf {
    path.map(PathBuf::from).unwrap_or_else(default_db_path)
}

fn native_error(code: &str, error: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(format!("RELAYHISTORY_NATIVE::{code}::{error}"))
}

fn worker_error(error: impl std::fmt::Display) -> napi::Error {
    native_error("WORKER_FAILED", error)
}

fn database_error(path: &Path, error: impl std::fmt::Display) -> napi::Error {
    native_error(
        "DATABASE_OPEN_FAILED",
        format!("could not open {}: {error}", path.display()),
    )
}

fn validate_limit(limit: Option<i64>, default: i64, max: i64) -> napi::Result<i64> {
    let limit = limit.unwrap_or(default);
    if !(1..=max).contains(&limit) {
        return Err(native_error(
            "INVALID_ARGUMENT",
            format!("limit must be between 1 and {max} (got {limit})"),
        ));
    }
    Ok(limit)
}

/// Contract version implemented by this native addon.
#[napi]
pub fn native_contract_version() -> u32 {
    NATIVE_CONTRACT_VERSION
}

/// Optimization profile this addon was compiled with: `release` or `debug`.
/// Performance measurements are only meaningful against `release`.
#[napi]
pub fn native_build_profile() -> String {
    if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        "release".to_string()
    }
}

#[napi(object)]
pub struct HistoryQueryOptions {
    pub db_path: Option<String>,
    pub source: Option<String>,
    pub project: Option<String>,
    pub tag: Option<String>,
    pub before_ms: Option<i64>,
    pub limit: Option<i64>,
}

impl HistoryQueryOptions {
    fn filter(&self, limit: i64) -> QueryFilter {
        QueryFilter {
            source: self.source.clone(),
            project: self.project.clone(),
            tag: self.tag.clone(),
            before_ms: self.before_ms,
            limit,
        }
    }
}

#[napi(object)]
pub struct SearchOptions {
    pub db_path: Option<String>,
    pub source: Option<String>,
    pub project: Option<String>,
    pub tag: Option<String>,
    pub before_ms: Option<i64>,
    pub limit: Option<i64>,
    pub raw_fts: Option<bool>,
}

#[napi(object)]
pub struct SessionOptions {
    pub db_path: Option<String>,
    pub source: Option<String>,
    pub tag: Option<String>,
}

#[napi(object)]
pub struct NativeHistoryEntry {
    pub id: i64,
    pub source: String,
    pub session_id: Option<String>,
    pub project: Option<String>,
    pub prompt: String,
    pub timestamp_ms: i64,
}

impl From<HistoryEntry> for NativeHistoryEntry {
    fn from(entry: HistoryEntry) -> Self {
        Self {
            id: entry.id,
            source: entry.source,
            session_id: entry.session_id,
            project: entry.project,
            prompt: entry.prompt,
            timestamp_ms: entry.timestamp_ms,
        }
    }
}

#[napi(object)]
pub struct NativeSessionEvent {
    pub id: i64,
    pub source: String,
    pub session_id: String,
    pub project: Option<String>,
    pub cwd: Option<String>,
    pub git_branch: Option<String>,
    pub message_id: Option<String>,
    pub parent_id: Option<String>,
    pub ts_ms: i64,
    pub role: String,
    pub kind: String,
    pub text: Option<String>,
    pub model: Option<String>,
    pub token_json: Option<String>,
    pub event_uid: String,
}

impl From<CoreSessionEvent> for NativeSessionEvent {
    fn from(event: CoreSessionEvent) -> Self {
        Self {
            id: event.id,
            source: event.source,
            session_id: event.session_id,
            project: event.project,
            cwd: event.cwd,
            git_branch: event.git_branch,
            message_id: event.message_id,
            parent_id: event.parent_id,
            ts_ms: event.ts_ms,
            role: event.role,
            kind: event.kind,
            text: event.text,
            model: event.model,
            token_json: event.token_json,
            event_uid: event.event_uid,
        }
    }
}

#[napi(object)]
pub struct EventCursor {
    pub ts_ms: i64,
    pub id: i64,
}

#[napi(object)]
pub struct EventsPageOptions {
    pub db_path: Option<String>,
    pub source: Option<String>,
    pub limit: Option<i64>,
    pub after: Option<EventCursor>,
}

#[napi(object)]
pub struct SessionEventsPage {
    pub events: Vec<NativeSessionEvent>,
    pub next_cursor: Option<EventCursor>,
}

#[napi(object)]
pub struct SourceCount {
    pub source: String,
    pub count: i64,
}

#[napi(object)]
pub struct ProjectCount {
    pub project: String,
    pub count: i64,
}

#[napi(object)]
pub struct NativeStats {
    pub total: i64,
    pub by_source: Vec<SourceCount>,
    pub by_project: Vec<ProjectCount>,
    pub first_timestamp_ms: Option<i64>,
    pub last_timestamp_ms: Option<i64>,
}

#[napi(object)]
pub struct StatsOptions {
    pub db_path: Option<String>,
    pub tag: Option<String>,
}

async fn read_database<T, F>(path: PathBuf, empty: T, operation: F) -> napi::Result<T>
where
    T: Send + 'static,
    F: FnOnce(&rusqlite::Connection) -> anyhow::Result<T> + Send + 'static,
{
    napi::tokio::task::spawn_blocking(move || {
        if !path.exists() {
            return Ok(empty);
        }
        let conn = match open_db_readonly(&path) {
            Ok(conn) if schema_is_current(&conn).unwrap_or(false) => conn,
            _ => open_db(&path).map_err(|error| database_error(&path, format!("{error:#}")))?,
        };
        operation(&conn)
            .map_err(|error| native_error("DATABASE_QUERY_FAILED", format!("{error:#}")))
    })
    .await
    .map_err(worker_error)?
}

/// Full-text search of indexed history. Never discovers or syncs implicitly.
#[napi]
pub async fn search(
    query: String,
    options: Option<SearchOptions>,
) -> napi::Result<Vec<NativeHistoryEntry>> {
    let options = options.unwrap_or(SearchOptions {
        db_path: None,
        source: None,
        project: None,
        tag: None,
        before_ms: None,
        limit: None,
        raw_fts: None,
    });
    let limit = validate_limit(options.limit, 20, 1_000)?;
    let path = db_path(options.db_path.clone());
    let filter = QueryFilter {
        source: options.source,
        project: options.project,
        tag: options.tag,
        before_ms: options.before_ms,
        limit,
    };
    let terms = query
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let raw_fts = options.raw_fts.unwrap_or(false);
    read_database(path, Vec::new(), move |conn| {
        Ok(core_search(conn, &terms, raw_fts, &filter)?
            .into_iter()
            .map(NativeHistoryEntry::from)
            .collect())
    })
    .await
}

/// Recent indexed history. Never discovers or syncs implicitly.
#[napi]
pub async fn recent(options: Option<HistoryQueryOptions>) -> napi::Result<Vec<NativeHistoryEntry>> {
    let options = options.unwrap_or(HistoryQueryOptions {
        db_path: None,
        source: None,
        project: None,
        tag: None,
        before_ms: None,
        limit: None,
    });
    let limit = validate_limit(options.limit, 20, 1_000)?;
    let path = db_path(options.db_path.clone());
    let filter = options.filter(limit);
    read_database(path, Vec::new(), move |conn| {
        Ok(core_recent(conn, &filter)?
            .into_iter()
            .map(NativeHistoryEntry::from)
            .collect())
    })
    .await
}

/// Indexed prompts for one session. Event payloads are separately paginated.
#[napi]
pub async fn get_session(
    session_id: String,
    options: Option<SessionOptions>,
) -> napi::Result<Vec<NativeHistoryEntry>> {
    let options = options.unwrap_or(SessionOptions {
        db_path: None,
        source: None,
        tag: None,
    });
    let path = db_path(options.db_path);
    read_database(path, Vec::new(), move |conn| {
        Ok(core_session(
            conn,
            &session_id,
            options.source.as_deref(),
            options.tag.as_deref(),
        )?
        .into_iter()
        .map(NativeHistoryEntry::from)
        .collect())
    })
    .await
}

/// One bounded page of normalized events for a session.
#[napi]
pub async fn get_session_events_page(
    session_id: String,
    options: Option<EventsPageOptions>,
) -> napi::Result<SessionEventsPage> {
    let options = options.unwrap_or(EventsPageOptions {
        db_path: None,
        source: None,
        limit: None,
        after: None,
    });
    let limit = validate_limit(options.limit, DEFAULT_EVENT_LIMIT, 1_000)?;
    let path = db_path(options.db_path);
    let after = options.after.map(|cursor| CoreEventCursor {
        ts_ms: cursor.ts_ms,
        id: cursor.id,
    });
    read_database(
        path,
        SessionEventsPage {
            events: Vec::new(),
            next_cursor: None,
        },
        move |conn| {
            let page = core_session_events_page(
                conn,
                &session_id,
                options.source.as_deref(),
                limit,
                after.as_ref(),
            )?;
            Ok(SessionEventsPage {
                events: page
                    .events
                    .into_iter()
                    .map(NativeSessionEvent::from)
                    .collect(),
                next_cursor: page.next_cursor.map(|cursor| EventCursor {
                    ts_ms: cursor.ts_ms,
                    id: cursor.id,
                }),
            })
        },
    )
    .await
}

/// Database statistics over already-indexed data.
#[napi]
pub async fn stats(options: Option<StatsOptions>) -> napi::Result<NativeStats> {
    let options = options.unwrap_or(StatsOptions {
        db_path: None,
        tag: None,
    });
    let path = db_path(options.db_path);
    read_database(
        path,
        NativeStats {
            total: 0,
            by_source: Vec::new(),
            by_project: Vec::new(),
            first_timestamp_ms: None,
            last_timestamp_ms: None,
        },
        move |conn| {
            let result = core_stats(conn, options.tag.as_deref())?;
            Ok(NativeStats {
                total: result.total,
                by_source: result
                    .by_source
                    .into_iter()
                    .map(|(source, count)| SourceCount { source, count })
                    .collect(),
                by_project: result
                    .by_project
                    .into_iter()
                    .map(|(project, count)| ProjectCount { project, count })
                    .collect(),
                first_timestamp_ms: result.first_timestamp_ms,
                last_timestamp_ms: result.last_timestamp_ms,
            })
        },
    )
    .await
}

#[napi(object)]
pub struct CatalogSession {
    pub source: String,
    pub session_id: String,
    pub cwd: Option<String>,
    pub git_branch: Option<String>,
    pub first_activity_ms: Option<i64>,
    pub last_activity_ms: Option<i64>,
    pub first_prompt: Option<String>,
    pub last_assistant_text: Option<String>,
    pub models: Vec<String>,
    pub originator: Option<String>,
    pub agent_version: Option<String>,
    pub repo_url: Option<String>,
    pub initial_commit: Option<String>,
    pub workspace_roots: Vec<String>,
    pub raw_path: Option<String>,
    pub source_stamp: Option<String>,
    pub discovery_state: String,
    pub from_cache: bool,
}

impl From<ai_hist_engine::ShallowSession> for CatalogSession {
    fn from(session: ai_hist_engine::ShallowSession) -> Self {
        Self {
            source: session.source,
            session_id: session.session_id,
            cwd: session.cwd,
            git_branch: session.git_branch,
            first_activity_ms: session.first_activity_ms,
            last_activity_ms: session.last_activity_ms,
            first_prompt: session.first_prompt,
            last_assistant_text: session.last_assistant_text,
            models: session.models,
            originator: session.originator,
            agent_version: session.agent_version,
            repo_url: session.repo_url,
            initial_commit: session.initial_commit,
            workspace_roots: session.workspace_roots,
            raw_path: session.raw_path,
            source_stamp: session.source_stamp,
            discovery_state: session.discovery_state,
            from_cache: session.from_cache,
        }
    }
}

#[napi(object)]
pub struct CatalogCursor {
    pub last_activity_ms: Option<i64>,
    pub source: String,
    pub session_id: String,
}

#[napi(object)]
pub struct ListCatalogOptions {
    pub db_path: Option<String>,
    pub sources: Option<Vec<String>>,
    pub limit: Option<i64>,
    pub before_ms: Option<i64>,
    pub after: Option<CatalogCursor>,
}

#[napi(object)]
pub struct SessionCatalogPage {
    pub contract_version: u32,
    pub sessions: Vec<CatalogSession>,
    pub next_cursor: Option<CatalogCursor>,
}

/// Cache-only indexed catalog query. Missing databases return an empty page.
#[napi]
pub async fn list_session_catalog_page(
    options: Option<ListCatalogOptions>,
) -> napi::Result<SessionCatalogPage> {
    let options = options.unwrap_or(ListCatalogOptions {
        db_path: None,
        sources: None,
        limit: None,
        before_ms: None,
        after: None,
    });
    validate_limit(options.limit, DEFAULT_LIMIT, 1_000)?;
    let path = db_path(options.db_path);
    let request = ai_hist_engine::CatalogListOptions {
        sources: options.sources.unwrap_or_default(),
        limit: options.limit,
        before_ms: options.before_ms,
        after: options.after.map(|cursor| ai_hist_engine::CatalogCursor {
            last_activity_ms: cursor.last_activity_ms,
            source: cursor.source,
            session_id: cursor.session_id,
        }),
    };
    napi::tokio::task::spawn_blocking(move || {
        ai_hist_engine::list_sessions_local_at(&path, &request)
    })
    .await
    .map_err(worker_error)?
    .map(|page| SessionCatalogPage {
        contract_version: ai_hist_engine::SESSION_CATALOG_CONTRACT_VERSION,
        sessions: page
            .sessions
            .into_iter()
            .map(CatalogSession::from)
            .collect(),
        next_cursor: page.next_cursor.map(|cursor| CatalogCursor {
            last_activity_ms: cursor.last_activity_ms,
            source: cursor.source,
            session_id: cursor.session_id,
        }),
    })
    .map_err(|error| native_error("DATABASE_QUERY_FAILED", format!("{error:#}")))
}

/// Convenience first-page catalog listing with identical cache-only semantics.
#[napi]
pub async fn list_session_catalog(
    options: Option<ListCatalogOptions>,
) -> napi::Result<Vec<CatalogSession>> {
    Ok(list_session_catalog_page(options).await?.sessions)
}

#[napi(object)]
pub struct DiscoverOptions {
    pub db_path: Option<String>,
    pub sources: Option<Vec<String>>,
    pub limit: Option<u32>,
}

#[napi(object)]
pub struct DiscoveryDiagnostic {
    pub source: String,
    pub locator: Option<String>,
    pub error: String,
}

#[napi(object)]
pub struct ProviderSummary {
    pub source: String,
    pub candidates: u32,
    pub discovered: u32,
    pub skipped_unchanged: u32,
    pub failed: bool,
}

#[napi(object)]
pub struct DiscoveryCounters {
    pub candidates_enumerated: i64,
    pub shallow_reads: i64,
    pub skipped_unchanged: i64,
    pub files_opened: i64,
    pub bytes_read: i64,
}

#[napi(object)]
pub struct SourceExemption {
    pub source: String,
    pub reason: String,
}

#[napi(object)]
pub struct DiscoverResult {
    pub contract_version: u32,
    pub sessions: Vec<CatalogSession>,
    pub discovered: u32,
    pub skipped_unchanged: u32,
    pub providers: Vec<ProviderSummary>,
    pub exempt_sources: Vec<SourceExemption>,
    pub diagnostics: Vec<DiscoveryDiagnostic>,
    pub counters: DiscoveryCounters,
}

/// Explicit bounded provider discovery. It updates only the session catalog.
#[napi]
pub async fn discover_sessions(options: Option<DiscoverOptions>) -> napi::Result<DiscoverResult> {
    let options = options.unwrap_or(DiscoverOptions {
        db_path: None,
        sources: None,
        limit: None,
    });
    let path = db_path(options.db_path);
    let request = ai_hist_engine::DiscoverOptions {
        sources: options.sources.unwrap_or_default(),
        limit: options.limit.map(|limit| limit as usize),
    };
    let (sessions, summary) = napi::tokio::task::spawn_blocking(move || {
        ai_hist_engine::discover_sessions_local_at(&path, &request)
    })
    .await
    .map_err(worker_error)?
    .map_err(|error| native_error("DISCOVERY_FAILED", format!("{error:#}")))?;
    Ok(DiscoverResult {
        contract_version: summary.contract_version,
        sessions: sessions.into_iter().map(CatalogSession::from).collect(),
        discovered: summary.discovered as u32,
        skipped_unchanged: summary.skipped_unchanged as u32,
        providers: summary
            .providers
            .into_iter()
            .map(|(source, provider)| ProviderSummary {
                source,
                candidates: provider.candidates as u32,
                discovered: provider.discovered as u32,
                skipped_unchanged: provider.skipped_unchanged as u32,
                failed: provider.failed,
            })
            .collect(),
        exempt_sources: summary
            .exempt_sources
            .into_iter()
            .map(|item| SourceExemption {
                source: item.source.to_string(),
                reason: item.reason.to_string(),
            })
            .collect(),
        diagnostics: summary
            .diagnostics
            .into_iter()
            .map(|item| DiscoveryDiagnostic {
                source: item.source,
                locator: item.locator,
                error: item.error,
            })
            .collect(),
        counters: DiscoveryCounters {
            candidates_enumerated: summary.counters.candidates_enumerated as i64,
            shallow_reads: summary.counters.shallow_reads as i64,
            skipped_unchanged: summary.counters.skipped_unchanged as i64,
            files_opened: summary.counters.files_opened as i64,
            bytes_read: summary.counters.bytes_read as i64,
        },
    })
}

#[napi(object)]
pub struct SyncOptions {
    pub db_path: Option<String>,
}

#[napi(object)]
pub struct SyncResult {
    pub database_path: String,
    pub completed: bool,
}

/// Explicit full ingestion. A read operation never calls this implicitly.
#[napi]
pub async fn sync(options: Option<SyncOptions>) -> napi::Result<SyncResult> {
    let path = db_path(options.and_then(|options| options.db_path));
    let result_path = path.display().to_string();
    napi::tokio::task::spawn_blocking(move || ai_hist_engine::sync_local_at(&path))
        .await
        .map_err(worker_error)?
        .map_err(|error| native_error("SYNC_FAILED", format!("{error:#}")))?;
    Ok(SyncResult {
        database_path: result_path,
        completed: true,
    })
}

/// Backward-compatible internal capture entry point; not used by the SDK.
#[napi]
pub async fn sync_local() -> napi::Result<()> {
    sync(None).await.map(|_| ())
}

#[napi(object)]
pub struct SyncPushResult {
    pub sent: u32,
    pub accepted: u32,
    pub authenticated: bool,
    pub sync_skipped: bool,
}

/// Cloud capture hook retained for Agent Relay integration.
#[napi]
pub async fn sync_and_push() -> napi::Result<SyncPushResult> {
    let outcome = napi::tokio::task::spawn_blocking(ai_hist_engine::sync_and_push)
        .await
        .map_err(worker_error)?
        .map_err(|error| native_error("SYNC_PUSH_FAILED", format!("{error:#}")))?;
    Ok(SyncPushResult {
        sent: outcome.sent as u32,
        accepted: outcome.accepted as u32,
        authenticated: outcome.authenticated,
        sync_skipped: outcome.sync_skipped,
    })
}
