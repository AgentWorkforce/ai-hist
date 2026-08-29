//! Native (napi) binding for driving ai-hist in-process — no CLI shell-out.
//!
//! Exposes local-only `syncLocal()` and cloud-enabled `syncAndPush()` to Node,
//! plus the session catalog: `listSessions()` (cache-only) and
//! `discoverSessions()` (shallow provider scan).
#![deny(clippy::all)]

use napi_derive::napi;

/// Result of one in-process sync+push.
#[napi(object)]
pub struct SyncPushResult {
    pub sent: u32,
    pub accepted: u32,
    /// `false` when there's no stored relayhistory auth yet (a no-op, not an error).
    pub authenticated: bool,
    /// `true` when another process owned the history scan; existing rows were still pushed.
    pub sync_skipped: bool,
}

/// Refresh local history without reading auth or pushing to cloud.
#[napi]
pub async fn sync_local() -> napi::Result<()> {
    napi::tokio::task::spawn_blocking(ai_hist_cli::sync_local)
        .await
        .map_err(|e| napi::Error::from_reason(format!("worker thread panicked: {e}")))?
        .map_err(|e| napi::Error::from_reason(format!("{e:#}")))
}

/// Sync local agent history into the ai-hist DB, then push new records to
/// relayhistory-cloud. The blocking work (file/SQLite/HTTP) runs on a worker
/// thread so the Node event loop is never blocked.
#[napi]
pub async fn sync_and_push() -> napi::Result<SyncPushResult> {
    let outcome = napi::tokio::task::spawn_blocking(ai_hist_cli::sync_and_push)
        .await
        .map_err(|e| napi::Error::from_reason(format!("worker thread panicked: {e}")))?
        .map_err(|e| napi::Error::from_reason(format!("{e:#}")))?;
    Ok(SyncPushResult {
        sent: outcome.sent as u32,
        accepted: outcome.accepted as u32,
        authenticated: outcome.authenticated,
        sync_skipped: outcome.sync_skipped,
    })
}

/// One coding-agent session as the shallow catalog knows it.
///
/// `firstPrompt` is derived by RelayHistory (a bounded excerpt of the first
/// substantive human turn); everything else is observed in provider data, and
/// anything the provider does not record stays null rather than being invented.
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
    /// `"shallow"` (catalog row only) or `"full"` (full evidence ingested).
    pub discovery_state: String,
    /// `true` when the row was served from the catalog without re-reading the
    /// provider source.
    pub from_cache: bool,
}

impl From<ai_hist_cli::ShallowSession> for CatalogSession {
    fn from(session: ai_hist_cli::ShallowSession) -> Self {
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

/// Filters for the cache-only catalog listing.
#[napi(object)]
pub struct ListSessionsOptions {
    /// Restrict to these sources. Omit for every discoverable source.
    pub sources: Option<Vec<String>>,
    /// Row cap (default 50).
    pub limit: Option<i64>,
    /// Keyset pagination: only sessions older than this epoch-ms cutoff.
    pub before_ms: Option<i64>,
}

/// The cache-only catalog listing plus the contract version it was built with.
#[napi(object)]
pub struct SessionCatalog {
    pub contract_version: u32,
    pub sessions: Vec<CatalogSession>,
}

/// Filters for one shallow discovery run.
#[napi(object)]
pub struct DiscoverSessionsOptions {
    /// Restrict to these sources. Omit for every adapter.
    pub sources: Option<Vec<String>>,
    /// Global cap across all providers, applied by recency. Omit for no cap.
    pub limit: Option<u32>,
}

/// A non-fatal failure during discovery. One provider (or one malformed
/// session) failing never blocks the rest of the run.
#[napi(object)]
pub struct DiscoveryDiagnostic {
    pub source: String,
    pub locator: Option<String>,
    pub error: String,
}

/// Per-provider tallies for one discovery run.
#[napi(object)]
pub struct ProviderSummary {
    pub source: String,
    pub candidates: u32,
    pub discovered: u32,
    pub skipped_unchanged: u32,
    pub failed: bool,
}

/// What one discovery run actually did — bounded-work evidence, not timings.
#[napi(object)]
pub struct DiscoveryCounters {
    pub candidates_enumerated: i64,
    pub shallow_reads: i64,
    pub skipped_unchanged: i64,
    pub files_opened: i64,
    pub bytes_read: i64,
}

/// A source that deliberately has no shallow adapter, and why.
#[napi(object)]
pub struct SourceExemption {
    pub source: String,
    pub reason: String,
}

/// Outcome of one shallow discovery run, with the rows collected.
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

/// List the session catalog from the local database only.
///
/// One indexed query over `sessions`: no provider transcript is opened and no
/// history/event/tool-call table is scanned, so this stays fast on first paint
/// even with thousands of historical sessions. A database that does not exist
/// yet is an empty catalog — call `discoverSessions()` to populate it.
#[napi]
pub async fn list_sessions(options: Option<ListSessionsOptions>) -> napi::Result<SessionCatalog> {
    let options = options.unwrap_or(ListSessionsOptions {
        sources: None,
        limit: None,
        before_ms: None,
    });
    let request = ai_hist_cli::CatalogListOptions {
        sources: options.sources.unwrap_or_default(),
        limit: options.limit,
        before_ms: options.before_ms,
    };
    let sessions =
        napi::tokio::task::spawn_blocking(move || ai_hist_cli::list_sessions_local(&request))
            .await
            .map_err(|e| napi::Error::from_reason(format!("worker thread panicked: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e:#}")))?;
    Ok(SessionCatalog {
        contract_version: ai_hist_cli::SESSION_CATALOG_CONTRACT_VERSION,
        sessions: sessions.into_iter().map(CatalogSession::from).collect(),
    })
}

/// Discover sessions from the known provider locations with bounded reads.
///
/// Candidates from every provider are merged by recency before the limit is
/// applied, so a limit is global rather than per-provider. Sources whose bytes
/// have not changed since the last run are served from the catalog. Rows are
/// collected rather than streamed; use the CLI's JSONL output for progressive
/// consumption.
#[napi]
pub async fn discover_sessions(
    options: Option<DiscoverSessionsOptions>,
) -> napi::Result<DiscoverResult> {
    let options = options.unwrap_or(DiscoverSessionsOptions {
        sources: None,
        limit: None,
    });
    let request = ai_hist_cli::DiscoverOptions {
        sources: options.sources.unwrap_or_default(),
        limit: options.limit.map(|limit| limit as usize),
    };
    let (sessions, summary) =
        napi::tokio::task::spawn_blocking(move || ai_hist_cli::discover_sessions_local(&request))
            .await
            .map_err(|e| napi::Error::from_reason(format!("worker thread panicked: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e:#}")))?;
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
            .map(|exemption| SourceExemption {
                source: exemption.source.to_string(),
                reason: exemption.reason.to_string(),
            })
            .collect(),
        diagnostics: summary
            .diagnostics
            .into_iter()
            .map(|diagnostic| DiscoveryDiagnostic {
                source: diagnostic.source,
                locator: diagnostic.locator,
                error: diagnostic.error,
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
