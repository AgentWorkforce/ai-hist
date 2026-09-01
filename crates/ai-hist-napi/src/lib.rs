//! Mandatory Node-API boundary for RelayHistory.
//!
//! This crate maps typed JavaScript requests to reusable Rust operations. It
//! contains no SQL, provider parsing, migration, or query semantics of its own.
#![deny(clippy::all)]

use std::path::{Path, PathBuf};

use ai_hist_core::{
    default_db_path, open_db, open_db_readonly, recent as core_recent,
    schema_is_catalog_read_current, schema_is_event_read_current, schema_is_evidence_read_current,
    schema_is_read_current, search as core_search, session as core_session,
    session_events_page as core_session_events_page,
    session_file_edits_page as core_session_file_edits_page, session_locations,
    session_tool_calls_page as core_session_tool_calls_page, stats_scoped as core_stats_scoped,
    HistoryEntry, QueryFilter, SessionEvent as CoreSessionEvent,
    SessionEventCursor as CoreEventCursor, SessionEvidenceCursor as CoreEvidenceCursor,
    SessionFileEdit as CoreSessionFileEdit, SessionScope, SessionToolCall as CoreSessionToolCall,
    SESSION_EVIDENCE_CONTRACT_VERSION,
};
use napi_derive::napi;

/// Bump whenever native object shapes or semantics require an SDK change.
pub const NATIVE_CONTRACT_VERSION: u32 = 5;
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

/// Session identity is two required strings; an empty one would silently widen
/// a page to every row a provider ever recorded.
fn validate_identity(value: String, field: &str) -> napi::Result<String> {
    if value.trim().is_empty() {
        return Err(native_error(
            "INVALID_ARGUMENT",
            format!("{field} must not be empty"),
        ));
    }
    Ok(value)
}

fn core_evidence_cursor(cursor: EvidenceCursor) -> CoreEvidenceCursor {
    CoreEvidenceCursor {
        ts_ms: cursor.ts_ms,
        id: cursor.id,
    }
}

fn evidence_cursor(cursor: CoreEvidenceCursor) -> EvidenceCursor {
    EvidenceCursor {
        ts_ms: cursor.ts_ms,
        id: cursor.id,
    }
}

fn parse_scope(scope: Option<String>) -> napi::Result<SessionScope> {
    match scope.as_deref().unwrap_or("local") {
        "local" => Ok(SessionScope::Local),
        "remote" => Ok(SessionScope::Remote),
        "all" => Ok(SessionScope::All),
        value => Err(native_error(
            "INVALID_ARGUMENT",
            format!("scope must be local, remote, or all (got '{value}')"),
        )),
    }
}

fn scope_name(scope: SessionScope) -> String {
    match scope {
        SessionScope::Local => "local",
        SessionScope::Remote => "remote",
        SessionScope::All => "all",
    }
    .to_string()
}

/// Reject remote-only acquisition that no configured remote connector can
/// serve — none configured at all, or a source filter that excludes every
/// configured one — keeping the SDK's stable `UNSUPPORTED_OPERATION`
/// taxonomy. The engine re-validates; this pre-check only classifies the
/// error.
fn ensure_acquisition_scope_supported(
    scope: SessionScope,
    operation: &str,
    sources: &[String],
) -> napi::Result<()> {
    if scope == SessionScope::Remote {
        ai_hist_engine::remote::ensure_remote_connectors_configured_for(operation, sources)
            .map_err(|error| native_error("UNSUPPORTED_OPERATION", format!("{error:#}")))?;
    }
    Ok(())
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
    pub scope: Option<String>,
    pub db_path: Option<String>,
    pub source: Option<String>,
    pub project: Option<String>,
    pub tag: Option<String>,
    pub before_ms: Option<i64>,
    pub limit: Option<i64>,
}

impl HistoryQueryOptions {
    fn filter(&self, limit: i64) -> napi::Result<QueryFilter> {
        Ok(QueryFilter {
            scope: parse_scope(self.scope.clone())?,
            source: self.source.clone(),
            project: self.project.clone(),
            tag: self.tag.clone(),
            before_ms: self.before_ms,
            limit,
        })
    }
}

#[napi(object)]
pub struct SearchOptions {
    pub scope: Option<String>,
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
    pub locations: Vec<String>,
}

impl NativeHistoryEntry {
    fn from_entry(conn: &rusqlite::Connection, entry: HistoryEntry) -> anyhow::Result<Self> {
        let locations = match entry.session_id.as_deref() {
            Some(session_id) => session_locations(conn, &entry.source, session_id)?,
            None => Vec::new(),
        };
        Ok(Self {
            id: entry.id,
            source: entry.source,
            session_id: entry.session_id,
            project: entry.project,
            prompt: entry.prompt,
            timestamp_ms: entry.timestamp_ms,
            locations,
        })
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
pub struct NativeSessionToolCall {
    pub id: i64,
    pub source: String,
    pub session_id: String,
    pub message_id: Option<String>,
    pub tool_use_id: String,
    pub name: String,
    pub target: Option<String>,
    /// Stored provider arguments, unparsed. The SDK owns JSON parsing so an
    /// unreadable value cannot fail a whole page at the boundary.
    pub args_json: Option<String>,
    pub is_error: Option<bool>,
    pub ts_ms: Option<i64>,
}

impl From<CoreSessionToolCall> for NativeSessionToolCall {
    fn from(call: CoreSessionToolCall) -> Self {
        Self {
            id: call.id,
            source: call.source,
            session_id: call.session_id,
            message_id: call.message_id,
            tool_use_id: call.tool_use_id,
            name: call.name,
            target: call.target,
            args_json: call.args_json,
            is_error: call.is_error.map(|value| value != 0),
            ts_ms: call.ts_ms,
        }
    }
}

#[napi(object)]
pub struct NativeSessionFileEdit {
    pub id: i64,
    pub source: String,
    pub session_id: String,
    pub message_id: Option<String>,
    pub tool_use_id: String,
    pub file_path: String,
    pub tool_name: Option<String>,
    pub lines_added: Option<i64>,
    pub lines_removed: Option<i64>,
    /// Stored provider patch, unparsed, for the same reason as `args_json`.
    pub structured_patch_json: Option<String>,
    pub user_modified: Option<bool>,
    pub ts_ms: Option<i64>,
    pub git_branch: Option<String>,
    pub cwd: Option<String>,
}

impl From<CoreSessionFileEdit> for NativeSessionFileEdit {
    fn from(edit: CoreSessionFileEdit) -> Self {
        Self {
            id: edit.id,
            source: edit.source,
            session_id: edit.session_id,
            message_id: edit.message_id,
            tool_use_id: edit.tool_use_id,
            file_path: edit.file_path,
            tool_name: edit.tool_name,
            lines_added: edit.lines_added,
            lines_removed: edit.lines_removed,
            structured_patch_json: edit.structured_patch_json,
            user_modified: edit.user_modified.map(|value| value != 0),
            ts_ms: edit.ts_ms,
            git_branch: edit.git_branch,
            cwd: edit.cwd,
        }
    }
}

/// Continuation for tool call and file edit pages. `tsMs` is nullable because
/// both tables order their undated rows last.
#[napi(object)]
pub struct EvidenceCursor {
    pub ts_ms: Option<i64>,
    pub id: i64,
}

#[napi(object)]
pub struct EvidencePageOptions {
    pub db_path: Option<String>,
    pub limit: Option<i64>,
    pub after: Option<EvidenceCursor>,
}

#[napi(object)]
pub struct SessionToolCallsPage {
    pub contract_version: u32,
    pub source: String,
    pub session_id: String,
    pub tool_calls: Vec<NativeSessionToolCall>,
    pub next_cursor: Option<EvidenceCursor>,
}

#[napi(object)]
pub struct SessionFileEditsPage {
    pub contract_version: u32,
    pub source: String,
    pub session_id: String,
    pub file_edits: Vec<NativeSessionFileEdit>,
    pub next_cursor: Option<EvidenceCursor>,
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
    pub scope: String,
    pub total: i64,
    pub by_source: Vec<SourceCount>,
    pub by_project: Vec<ProjectCount>,
    pub first_timestamp_ms: Option<i64>,
    pub last_timestamp_ms: Option<i64>,
}

#[napi(object)]
pub struct StatsOptions {
    pub scope: Option<String>,
    pub db_path: Option<String>,
    pub tag: Option<String>,
}

async fn read_database<T, F>(path: PathBuf, empty: T, operation: F) -> napi::Result<T>
where
    T: Send + 'static,
    F: FnOnce(&rusqlite::Connection) -> anyhow::Result<T> + Send + 'static,
{
    read_database_with_schema(path, empty, schema_is_read_current, operation).await
}

async fn read_database_with_schema<T, F>(
    path: PathBuf,
    empty: T,
    schema_current: fn(&rusqlite::Connection) -> anyhow::Result<bool>,
    operation: F,
) -> napi::Result<T>
where
    T: Send + 'static,
    F: FnOnce(&rusqlite::Connection) -> anyhow::Result<T> + Send + 'static,
{
    napi::tokio::task::spawn_blocking(move || {
        if !path.exists() {
            return Ok(empty);
        }
        let conn = match open_db_readonly(&path) {
            Ok(conn) if schema_current(&conn).unwrap_or(false) => conn,
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
        scope: None,
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
        scope: parse_scope(options.scope)?,
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
        core_search(conn, &terms, raw_fts, &filter)?
            .into_iter()
            .map(|entry| NativeHistoryEntry::from_entry(conn, entry))
            .collect()
    })
    .await
}

/// Recent indexed history. Never discovers or syncs implicitly.
#[napi]
pub async fn recent(options: Option<HistoryQueryOptions>) -> napi::Result<Vec<NativeHistoryEntry>> {
    let options = options.unwrap_or(HistoryQueryOptions {
        scope: None,
        db_path: None,
        source: None,
        project: None,
        tag: None,
        before_ms: None,
        limit: None,
    });
    let limit = validate_limit(options.limit, 20, 1_000)?;
    let path = db_path(options.db_path.clone());
    let filter = options.filter(limit)?;
    read_database(path, Vec::new(), move |conn| {
        core_recent(conn, &filter)?
            .into_iter()
            .map(|entry| NativeHistoryEntry::from_entry(conn, entry))
            .collect()
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
        core_session(
            conn,
            &session_id,
            options.source.as_deref(),
            options.tag.as_deref(),
        )?
        .into_iter()
        .map(|entry| NativeHistoryEntry::from_entry(conn, entry))
        .collect()
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
    read_database_with_schema(
        path,
        SessionEventsPage {
            events: Vec::new(),
            next_cursor: None,
        },
        schema_is_event_read_current,
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

/// One bounded page of recorded tool calls for one session.
///
/// Both `source` and `sessionId` are required: provider session ids are not
/// globally unique, so an id-only lookup could interleave two sessions.
#[napi]
pub async fn get_session_tool_calls_page(
    source: String,
    session_id: String,
    options: Option<EvidencePageOptions>,
) -> napi::Result<SessionToolCallsPage> {
    let options = options.unwrap_or(EvidencePageOptions {
        db_path: None,
        limit: None,
        after: None,
    });
    let source = validate_identity(source, "source")?;
    let session_id = validate_identity(session_id, "sessionId")?;
    let limit = validate_limit(options.limit, DEFAULT_EVENT_LIMIT, 1_000)?;
    let path = db_path(options.db_path);
    let after = options.after.map(core_evidence_cursor);
    let (page_source, page_session_id) = (source.clone(), session_id.clone());
    read_database_with_schema(
        path,
        ai_hist_core::SessionToolCallPage {
            tool_calls: Vec::new(),
            next_cursor: None,
        },
        schema_is_evidence_read_current,
        move |conn| core_session_tool_calls_page(conn, &source, &session_id, limit, after.as_ref()),
    )
    .await
    .map(|page| SessionToolCallsPage {
        contract_version: SESSION_EVIDENCE_CONTRACT_VERSION,
        source: page_source,
        session_id: page_session_id,
        tool_calls: page
            .tool_calls
            .into_iter()
            .map(NativeSessionToolCall::from)
            .collect(),
        next_cursor: page.next_cursor.map(evidence_cursor),
    })
}

/// One bounded page of recorded file edits for one session.
#[napi]
pub async fn get_session_file_edits_page(
    source: String,
    session_id: String,
    options: Option<EvidencePageOptions>,
) -> napi::Result<SessionFileEditsPage> {
    let options = options.unwrap_or(EvidencePageOptions {
        db_path: None,
        limit: None,
        after: None,
    });
    let source = validate_identity(source, "source")?;
    let session_id = validate_identity(session_id, "sessionId")?;
    let limit = validate_limit(options.limit, DEFAULT_EVENT_LIMIT, 1_000)?;
    let path = db_path(options.db_path);
    let after = options.after.map(core_evidence_cursor);
    let (page_source, page_session_id) = (source.clone(), session_id.clone());
    read_database_with_schema(
        path,
        ai_hist_core::SessionFileEditPage {
            file_edits: Vec::new(),
            next_cursor: None,
        },
        schema_is_evidence_read_current,
        move |conn| core_session_file_edits_page(conn, &source, &session_id, limit, after.as_ref()),
    )
    .await
    .map(|page| SessionFileEditsPage {
        contract_version: SESSION_EVIDENCE_CONTRACT_VERSION,
        source: page_source,
        session_id: page_session_id,
        file_edits: page
            .file_edits
            .into_iter()
            .map(NativeSessionFileEdit::from)
            .collect(),
        next_cursor: page.next_cursor.map(evidence_cursor),
    })
}

/// Database statistics over already-indexed data.
#[napi]
pub async fn stats(options: Option<StatsOptions>) -> napi::Result<NativeStats> {
    let options = options.unwrap_or(StatsOptions {
        scope: None,
        db_path: None,
        tag: None,
    });
    let scope = parse_scope(options.scope)?;
    let path = db_path(options.db_path);
    read_database(
        path,
        NativeStats {
            scope: scope_name(scope),
            total: 0,
            by_source: Vec::new(),
            by_project: Vec::new(),
            first_timestamp_ms: None,
            last_timestamp_ms: None,
        },
        move |conn| {
            let result = core_stats_scoped(conn, options.tag.as_deref(), scope)?;
            Ok(NativeStats {
                scope: scope_name(scope),
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
    pub locations: Vec<String>,
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
            locations: session.locations,
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
    pub scope: Option<String>,
    pub db_path: Option<String>,
    pub sources: Option<Vec<String>>,
    pub limit: Option<i64>,
    pub before_ms: Option<i64>,
    pub after: Option<CatalogCursor>,
}

#[napi(object)]
pub struct SessionCatalogPage {
    pub contract_version: u32,
    pub scope: String,
    pub sessions: Vec<CatalogSession>,
    pub next_cursor: Option<CatalogCursor>,
}

/// Cache-only indexed catalog query. Missing databases return an empty page.
#[napi]
pub async fn list_session_catalog_page(
    options: Option<ListCatalogOptions>,
) -> napi::Result<SessionCatalogPage> {
    let options = options.unwrap_or(ListCatalogOptions {
        scope: None,
        db_path: None,
        sources: None,
        limit: None,
        before_ms: None,
        after: None,
    });
    validate_limit(options.limit, DEFAULT_LIMIT, 1_000)?;
    let scope = parse_scope(options.scope)?;
    let path = db_path(options.db_path);
    let request = ai_hist_engine::CatalogListOptions {
        scope,
        sources: options.sources.unwrap_or_default(),
        limit: options.limit,
        before_ms: options.before_ms,
        after: options.after.map(|cursor| ai_hist_engine::CatalogCursor {
            last_activity_ms: cursor.last_activity_ms,
            source: cursor.source,
            session_id: cursor.session_id,
        }),
    };
    read_database_with_schema(
        path,
        ai_hist_engine::SessionCatalogPage {
            scope,
            ..Default::default()
        },
        schema_is_catalog_read_current,
        move |conn| ai_hist_engine::list_session_catalog_page(conn, &request),
    )
    .await
    .map(|page| SessionCatalogPage {
        contract_version: ai_hist_engine::SESSION_CATALOG_CONTRACT_VERSION,
        scope: scope_name(page.scope),
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
    pub scope: Option<String>,
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
    pub scope: String,
    pub locations_run: Vec<String>,
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
        scope: None,
        db_path: None,
        sources: None,
        limit: None,
    });
    let scope = parse_scope(options.scope)?;
    let sources = options.sources.unwrap_or_default();
    ensure_acquisition_scope_supported(scope, "discovery", &sources)?;
    let path = db_path(options.db_path);
    let request = ai_hist_engine::DiscoverOptions {
        scope,
        sources,
        limit: options.limit.map(|limit| limit as usize),
    };
    let (sessions, summary) = napi::tokio::task::spawn_blocking(move || {
        ai_hist_engine::discover_sessions_scoped_at(&path, &request)
    })
    .await
    .map_err(worker_error)?
    .map_err(|error| native_error("DISCOVERY_FAILED", format!("{error:#}")))?;
    Ok(DiscoverResult {
        contract_version: summary.contract_version,
        scope: scope_name(scope),
        locations_run: summary.locations_run,
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
pub struct HydrateSessionOptions {
    pub source: String,
    pub session_id: String,
    pub scope: Option<String>,
    pub db_path: Option<String>,
    pub include_related: Option<bool>,
}

#[napi(object)]
pub struct HydrationIndexedThrough {
    pub source_stamp: Option<String>,
    pub last_event_at_ms: Option<i64>,
}

#[napi(object)]
pub struct HydrationEvidence {
    pub prompts: i64,
    pub events: i64,
    pub tool_calls: i64,
    pub related_sessions: i64,
}

#[napi(object)]
pub struct HydrationDiagnostic {
    pub code: String,
    pub message: String,
    pub duration_ms: Option<i64>,
    pub source_bytes: Option<i64>,
    pub records_parsed: Option<i64>,
}

#[napi(object)]
pub struct HydrateSessionResult {
    pub contract_version: u32,
    pub source: String,
    pub session_id: String,
    pub status: String,
    pub discovery_state: String,
    pub presence: String,
    pub indexed_through: HydrationIndexedThrough,
    pub evidence: HydrationEvidence,
    pub related_session_ids: Vec<String>,
    pub diagnostics: Vec<HydrationDiagnostic>,
}

fn hydration_error(error: anyhow::Error) -> napi::Error {
    let message = format!("{error:#}");
    for code in [
        "SESSION_NOT_FOUND",
        "SESSION_SOURCE_UNAVAILABLE",
        "SESSION_SOURCE_MISMATCH",
        "HYDRATION_UNSUPPORTED",
        "INVALID_ARGUMENT",
    ] {
        if let Some(detail) = message.strip_prefix(&format!("{code}: ")) {
            return native_error(code, detail);
        }
    }
    native_error("HYDRATION_FAILED", message)
}

/// Fully index one cataloged session without enumerating unrelated sessions.
#[napi]
pub async fn hydrate_session(options: HydrateSessionOptions) -> napi::Result<HydrateSessionResult> {
    let scope = parse_scope(options.scope)?;
    let path = db_path(options.db_path);
    let request = ai_hist_engine::HydrateSessionOptions {
        source: options.source,
        session_id: options.session_id,
        scope,
        include_related: options.include_related.unwrap_or(true),
    };
    let result = napi::tokio::task::spawn_blocking(move || {
        ai_hist_engine::hydrate_session_at(&path, &request)
    })
    .await
    .map_err(worker_error)?
    .map_err(hydration_error)?;
    Ok(HydrateSessionResult {
        contract_version: result.contract_version,
        source: result.source,
        session_id: result.session_id,
        status: result.status,
        discovery_state: result.discovery_state,
        presence: result.presence,
        indexed_through: HydrationIndexedThrough {
            source_stamp: result.indexed_through.source_stamp,
            last_event_at_ms: result.indexed_through.last_event_at_ms,
        },
        evidence: HydrationEvidence {
            prompts: result.evidence.prompts as i64,
            events: result.evidence.events as i64,
            tool_calls: result.evidence.tool_calls as i64,
            related_sessions: result.evidence.related_sessions as i64,
        },
        related_session_ids: result.related_session_ids,
        diagnostics: result
            .diagnostics
            .into_iter()
            .map(|diagnostic| HydrationDiagnostic {
                code: diagnostic.code,
                message: diagnostic.message,
                duration_ms: diagnostic.duration_ms,
                source_bytes: diagnostic.source_bytes,
                records_parsed: diagnostic.records_parsed,
            })
            .collect(),
    })
}

#[napi(object)]
pub struct SyncOptions {
    pub db_path: Option<String>,
    pub scope: Option<String>,
}

#[napi(object)]
pub struct SyncResult {
    pub database_path: String,
    pub completed: bool,
    pub scope: String,
}

/// Explicit full ingestion. A read operation never calls this implicitly.
#[napi]
pub async fn sync(options: Option<SyncOptions>) -> napi::Result<SyncResult> {
    let options = options.unwrap_or(SyncOptions {
        db_path: None,
        scope: None,
    });
    let scope = parse_scope(options.scope)?;
    ensure_acquisition_scope_supported(scope, "sync", &[])?;
    let path = db_path(options.db_path);
    let result_path = path.display().to_string();
    let completed =
        napi::tokio::task::spawn_blocking(move || ai_hist_engine::sync_scoped_at(&path, scope))
            .await
            .map_err(worker_error)?
            .map_err(|error| native_error("SYNC_FAILED", format!("{error:#}")))?;
    Ok(SyncResult {
        database_path: result_path,
        completed,
        scope: scope_name(scope),
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
