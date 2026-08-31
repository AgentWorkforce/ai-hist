//! Remote session connectors: claude.ai/code web sessions and Codex cloud tasks.
//!
//! A remote connector enumerates the sessions a provider keeps on its own
//! service — Claude Code sessions running on claude.ai/code, Codex cloud
//! tasks — and lands them in the shared session ledger as catalog rows with a
//! `remote` presence. Connectors plug into the same shallow-discovery engine
//! as the file-backed local adapters (see [`crate::ShallowSessionProvider`]);
//! the network listing *is* the enumeration, and the per-candidate "read" is
//! served from the rows that listing already carried.
//!
//! # Capability boundary
//!
//! A connector is **configured** when the provider's own CLI has been signed
//! in on this machine — RelayHistory never runs an auth flow of its own:
//!
//! * `claude-web` — `~/.claude/.credentials.json` holds the claude.ai OAuth
//!   token the Claude Code CLI stored at sign-in (override the path with
//!   `RELAYHISTORY_CLAUDE_CREDENTIALS` for setups that keep credentials
//!   elsewhere, e.g. macOS keychain users who export a token file).
//! * `codex-cloud` — `~/.codex/auth.json` exists (written by `codex login`),
//!   and the `codex` CLI is invoked for the actual listing.
//!
//! Requesting `--remote` acquisition with no connector configured fails
//! loudly, exactly as before connectors existed; `--all` runs whatever is
//! configured and never errors on absence.
//!
//! # Fidelity
//!
//! Remote listings carry less than a local transcript, and nothing is
//! invented to fill the gap. The provider's session/task *title* is stored as
//! `first_prompt` — for both providers the title is derived from the opening
//! prompt, and it is the only human-readable identifier the listing offers.
//! Remote rows stay `discovery_state = "shallow"`; full remote transcript
//! ingestion is a separate capability that has not shipped.
//!
//! # Contract stability
//!
//! `codex-cloud` speaks the Codex CLI's documented scripting contract
//! (`codex cloud list --json`). `claude-web` speaks the same session-list
//! endpoint the Claude Code CLI's `--teleport` picker uses
//! (`GET {base}/v1/code/sessions`); that endpoint is **not documented as a
//! public API** and can change with any Claude Code release, so failures are
//! reported as connector diagnostics, never silently swallowed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ai_hist_core::SessionLocation;
use anyhow::{Context, Result};
use rusqlite::Connection;
use serde_json::Value;

use crate::discover::{Candidate, DiscoveryEnv, ScanEnv, ShallowSession, ShallowSessionProvider};

/// Connector name for the claude.ai/code web-session lister.
pub const CLAUDE_WEB_CONNECTOR: &str = "claude-web";
/// Connector name for the Codex cloud task lister.
pub const CODEX_CLOUD_CONNECTOR: &str = "codex-cloud";

/// Most listing pages one enumeration may fetch, whatever the caller asked.
const MAX_LIST_PAGES: usize = 100;
/// Rows requested per claude.ai listing page (the endpoint's own maximum).
const CLAUDE_PAGE_LIMIT: usize = 100;

/// Whether one remote connector can run on this machine, and why not when it
/// cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteConnectorStatus {
    /// Connector name (`claude-web`, `codex-cloud`).
    pub connector: &'static str,
    /// The `SOURCE_CHOICES` source its sessions land under.
    pub source: &'static str,
    /// `true` when the provider CLI's stored sign-in was found.
    pub configured: bool,
    /// Human-readable detail: the credential looked for, or where it was found.
    pub detail: String,
}

fn claude_credentials_path(home: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("RELAYHISTORY_CLAUDE_CREDENTIALS") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    home.join(".claude/.credentials.json")
}

fn codex_auth_path(home: &Path) -> PathBuf {
    home.join(".codex/auth.json")
}

/// Report each connector's availability under an explicit home directory.
pub fn remote_connector_statuses_at(home: &Path) -> Vec<RemoteConnectorStatus> {
    let claude_path = claude_credentials_path(home);
    let codex_path = codex_auth_path(home);
    vec![
        RemoteConnectorStatus {
            connector: CLAUDE_WEB_CONNECTOR,
            source: "claude",
            configured: claude_path.is_file(),
            detail: if claude_path.is_file() {
                format!("claude.ai credentials at {}", claude_path.display())
            } else {
                format!(
                    "no claude.ai credentials at {} (sign in with the Claude Code CLI)",
                    claude_path.display()
                )
            },
        },
        RemoteConnectorStatus {
            connector: CODEX_CLOUD_CONNECTOR,
            source: "codex",
            configured: codex_path.is_file(),
            detail: if codex_path.is_file() {
                format!("Codex CLI login at {}", codex_path.display())
            } else {
                format!(
                    "no Codex CLI login at {} (run `codex login`)",
                    codex_path.display()
                )
            },
        },
    ]
}

/// [`remote_connector_statuses_at`] under the process home directory.
pub fn remote_connector_statuses() -> Vec<RemoteConnectorStatus> {
    remote_connector_statuses_at(&crate::home_dir())
}

/// The error a remote-only acquisition request gets when nothing is configured.
///
/// The leading phrase is a compatibility contract: callers and tests match on
/// "no remote provider connectors are configured".
pub(crate) fn unconfigured_message(operation: &str, statuses: &[RemoteConnectorStatus]) -> String {
    let reasons = statuses
        .iter()
        .map(|status| format!("{}: {}", status.connector, status.detail))
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "remote session {operation} is not available: no remote provider connectors are configured ({reasons})"
    )
}

/// Reject a remote-only acquisition when no connector is configured, using the
/// process home directory. Cheap (a couple of `stat`s), so callers run it
/// before opening the ledger.
pub fn ensure_remote_connectors_configured(operation: &str) -> Result<()> {
    ensure_remote_connectors_configured_at(operation, &crate::home_dir())
}

/// [`ensure_remote_connectors_configured`] under an explicit home directory.
pub fn ensure_remote_connectors_configured_at(operation: &str, home: &Path) -> Result<()> {
    ensure_remote_connectors_configured_for_at(operation, home, &[])
}

/// Reject a remote-only acquisition that no configured connector can serve
/// once a source filter is applied, using the process home directory.
///
/// An empty `sources` filter means "every source". A filter that names only
/// sources without a remote connector (or whose connectors are not signed
/// in) is the same unsupported request, scoped down — callers classify both
/// as unsupported-operation, never as a runtime discovery failure.
pub fn ensure_remote_connectors_configured_for(operation: &str, sources: &[String]) -> Result<()> {
    ensure_remote_connectors_configured_for_at(operation, &crate::home_dir(), sources)
}

/// [`ensure_remote_connectors_configured_for`] under an explicit home directory.
pub fn ensure_remote_connectors_configured_for_at(
    operation: &str,
    home: &Path,
    sources: &[String],
) -> Result<()> {
    let statuses: Vec<RemoteConnectorStatus> = remote_connector_statuses_at(home)
        .into_iter()
        .filter(|status| sources.is_empty() || sources.iter().any(|s| s == status.source))
        .collect();
    anyhow::ensure!(
        !statuses.is_empty(),
        "remote session {operation} is not available for the requested source(s): no matching remote provider connectors exist"
    );
    anyhow::ensure!(
        statuses.iter().any(|status| status.configured),
        unconfigured_message(operation, &statuses)
    );
    Ok(())
}

/// Every configured remote connector, as shallow providers the discovery
/// engine can run beside the local adapters. Unconfigured connectors are
/// simply absent — for `all` scope that is the documented "runs whatever is
/// available" behaviour, and for `remote` scope the caller has already
/// rejected the empty set.
pub(crate) fn configured_remote_providers(
    home: &Path,
    limit: Option<usize>,
) -> Vec<Box<dyn ShallowSessionProvider>> {
    let mut providers: Vec<Box<dyn ShallowSessionProvider>> = Vec::new();
    let claude_path = claude_credentials_path(home);
    if claude_path.is_file() {
        providers.push(Box::new(ClaudeWebProvider::new(
            claude_path,
            claude_api_base_url(),
            Box::new(UreqClaudeTransport),
            limit,
        )));
    }
    let codex_path = codex_auth_path(home);
    if codex_path.is_file() {
        providers.push(Box::new(CodexCloudProvider::new(
            Box::new(ExecCodexCli),
            limit,
        )));
    }
    providers
}

// ---------------------------------------------------------------------------
// shared plumbing
// ---------------------------------------------------------------------------

/// Rows one remote enumeration fetched, keyed by candidate locator, waiting
/// for the engine's `read_shallow` calls. The listing already carried every
/// field, so the "read" is a map lookup.
type FetchedRows = Mutex<BTreeMap<String, ShallowSession>>;

fn take_fetched(rows: &FetchedRows, locator: &str) -> Option<ShallowSession> {
    rows.lock()
        .expect("remote connector row cache")
        .get(locator)
        .cloned()
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// claude-web
// ---------------------------------------------------------------------------

/// One HTTP response from the claude.ai session-list endpoint.
pub(crate) struct ClaudeHttpResponse {
    pub status: u16,
    pub body: String,
}

/// The HTTP side of the claude.ai session listing, abstracted so mapping and
/// pagination are testable without a network.
pub(crate) trait ClaudeSessionsTransport: Send + Sync {
    fn get(&self, url: &str, bearer_token: &str) -> Result<ClaudeHttpResponse>;
}

struct UreqClaudeTransport;

impl ClaudeSessionsTransport for UreqClaudeTransport {
    fn get(&self, url: &str, bearer_token: &str) -> Result<ClaudeHttpResponse> {
        let request = ureq::get(url)
            .timeout(std::time::Duration::from_secs(30))
            .set("Authorization", &format!("Bearer {bearer_token}"))
            .set("Content-Type", "application/json")
            .set("anthropic-version", "2023-06-01")
            .set("anthropic-beta", "oauth-2025-04-20");
        match request.call() {
            Ok(response) => Ok(ClaudeHttpResponse {
                status: response.status(),
                body: response.into_string().unwrap_or_default(),
            }),
            Err(ureq::Error::Status(status, response)) => Ok(ClaudeHttpResponse {
                status,
                body: response.into_string().unwrap_or_default(),
            }),
            Err(error) => Err(anyhow::Error::from(error).context("claude.ai session list request")),
        }
    }
}

fn claude_api_base_url() -> String {
    // Deliberately NOT `ANTHROPIC_BASE_URL`: that variable redirects generic
    // Anthropic API traffic (LLM gateways, dev proxies), and following it here
    // would hand the claude.ai OAuth token to whatever host it happens to
    // name. Redirecting the session-list endpoint — and with it the stored
    // credential — must be its own explicit decision.
    let raw = std::env::var("RELAYHISTORY_CLAUDE_API_BASE_URL").unwrap_or_default();
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        "https://api.anthropic.com".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Reject a plaintext listing endpoint. Loopback is exempt so a local mock
/// (or proxy) works without ceremony — the same posture `cloud.rs` takes for
/// `wrangler dev`.
fn require_https_or_loopback(base_url: &str) -> Result<()> {
    if base_url.starts_with("https://") {
        return Ok(());
    }
    let rest = base_url
        .strip_prefix("http://")
        .with_context(|| format!("claude.ai base URL must be http(s), got {base_url}"))?;
    let authority = rest.split('/').next().unwrap_or_default();
    let host = authority
        .rsplit('@')
        .next()
        .unwrap_or_default()
        .split(':')
        .next()
        .unwrap_or_default();
    anyhow::ensure!(
        matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1"),
        "refusing to send the claude.ai OAuth token over plain http:// to {base_url}; use an https:// endpoint (plain http is accepted only for loopback)"
    );
    Ok(())
}

/// The claude.ai OAuth token as the Claude Code CLI stores it.
struct ClaudeOauth {
    access_token: String,
    expires_at_ms: Option<i64>,
}

fn load_claude_oauth(path: &Path) -> Result<ClaudeOauth> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("could not read claude.ai credentials at {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("claude.ai credentials at {} are not JSON", path.display()))?;
    let oauth = value
        .get("claudeAiOauth")
        .with_context(|| format!("no claudeAiOauth entry in {}", path.display()))?;
    let access_token = string_field(oauth, "accessToken")
        .with_context(|| format!("no claudeAiOauth.accessToken in {}", path.display()))?;
    Ok(ClaudeOauth {
        access_token,
        expires_at_ms: oauth.get("expiresAt").and_then(Value::as_i64),
    })
}

/// Is this a claude.ai code-session id (`session_…` / `cse_…`)?
fn is_claude_web_session_id(id: &str) -> bool {
    let rest = id
        .strip_prefix("session_")
        .or_else(|| id.strip_prefix("cse_"));
    match rest {
        Some(rest) if !rest.is_empty() => rest
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'),
        _ => false,
    }
}

/// Map one session object from `GET /v1/code/sessions` into a catalog row,
/// or `None` when the entry is not a remote coding session we should index
/// (a malformed id, or a Remote Control bridge mirroring a *local* session).
fn map_claude_web_session(value: &Value) -> Option<(Candidate, ShallowSession)> {
    let id = string_field(value, "id").filter(|id| is_claude_web_session_id(id))?;
    // `environment_kind: "bridge"` is a Remote Control view of a session that
    // runs in a local terminal. Its evidence is local, not remote; the local
    // transcript adapter already covers it.
    if string_field(value, "environment_kind").as_deref() == Some("bridge") {
        return None;
    }
    let created_at = string_field(value, "created_at");
    let last_event_at = string_field(value, "last_event_at");
    let first_activity_ms = created_at.as_deref().and_then(crate::parse_iso_ms);
    let last_activity_ms = last_event_at
        .as_deref()
        .and_then(crate::parse_iso_ms)
        .or(first_activity_ms);
    let repo_url = value
        .pointer("/config/sources")
        .and_then(Value::as_array)
        .and_then(|sources| {
            sources
                .iter()
                .find(|source| string_field(source, "type").as_deref() == Some("git_repository"))
        })
        .and_then(|source| string_field(source, "url"));
    let stamp = format!(
        "web:{}",
        last_event_at
            .or(created_at)
            .unwrap_or_else(|| "unknown".to_string())
    );
    let session = ShallowSession {
        source: "claude".into(),
        session_id: id.clone(),
        // The listing's title is the only human-readable identifier the
        // endpoint offers; claude.ai derives it from the opening prompt.
        first_prompt: string_field(value, "title"),
        first_activity_ms,
        last_activity_ms,
        repo_url,
        raw_path: Some(format!("https://claude.ai/code/{id}")),
        discovery_state: "shallow".into(),
        ..Default::default()
    };
    let candidate = Candidate {
        source: "claude",
        locator: id.clone(),
        session_id: Some(id),
        recency_hint_ms: last_activity_ms,
        stamp,
    };
    Some((candidate, session))
}

/// Shallow adapter over the claude.ai/code session list.
pub(crate) struct ClaudeWebProvider {
    credentials_path: PathBuf,
    base_url: String,
    transport: Box<dyn ClaudeSessionsTransport>,
    limit: Option<usize>,
    fetched: FetchedRows,
}

impl ClaudeWebProvider {
    pub(crate) fn new(
        credentials_path: PathBuf,
        base_url: String,
        transport: Box<dyn ClaudeSessionsTransport>,
        limit: Option<usize>,
    ) -> Self {
        Self {
            credentials_path,
            base_url,
            transport,
            limit,
            fetched: FetchedRows::default(),
        }
    }
}

impl ShallowSessionProvider for ClaudeWebProvider {
    fn source(&self) -> &'static str {
        "claude"
    }

    fn location(&self) -> SessionLocation {
        SessionLocation::Remote
    }

    fn enumerate(&self, _env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        require_https_or_loopback(&self.base_url)?;
        let oauth = load_claude_oauth(&self.credentials_path)?;
        if let Some(expires_at_ms) = oauth.expires_at_ms {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis() as i64)
                .unwrap_or_default();
            anyhow::ensure!(
                expires_at_ms > now_ms,
                "the stored claude.ai OAuth token has expired; run the Claude Code CLI once to refresh it"
            );
        }
        let mut candidates = Vec::new();
        let mut rows = BTreeMap::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let page_limit = match self.limit {
                Some(limit) => limit
                    .saturating_sub(candidates.len())
                    .clamp(1, CLAUDE_PAGE_LIMIT),
                None => CLAUDE_PAGE_LIMIT,
            };
            let mut url = format!("{}/v1/code/sessions?limit={page_limit}", self.base_url);
            if let Some(cursor) = cursor.as_deref() {
                url.push_str("&cursor=");
                url.push_str(&urlencode(cursor));
            }
            let response = self.transport.get(&url, &oauth.access_token)?;
            match response.status {
                200 => {}
                401 | 403 => anyhow::bail!(
                    "claude.ai rejected the stored OAuth token (HTTP {}); run the Claude Code CLI once to refresh your sign-in",
                    response.status
                ),
                status => anyhow::bail!(
                    "claude.ai session list failed (HTTP {status}): {}",
                    excerpt_one_line(&response.body)
                ),
            }
            let payload: Value = serde_json::from_str(&response.body)
                .context("claude.ai session list returned unparseable JSON")?;
            let page = payload
                .get("data")
                .and_then(Value::as_array)
                .context("claude.ai session list response has no data array")?;
            for entry in page {
                if let Some((candidate, session)) = map_claude_web_session(entry) {
                    rows.insert(candidate.locator.clone(), session);
                    candidates.push(candidate);
                }
            }
            cursor = string_field(&payload, "next_cursor");
            let done = cursor.is_none()
                || page.is_empty()
                || self.limit.is_some_and(|limit| candidates.len() >= limit);
            if done {
                break;
            }
        }
        *self.fetched.lock().expect("remote connector row cache") = rows;
        Ok(candidates)
    }

    fn read_shallow(
        &self,
        _scan: &ScanEnv<'_>,
        _catalog: Option<&Connection>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        Ok(take_fetched(&self.fetched, &candidate.locator))
    }
}

fn urlencode(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for byte in raw.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn excerpt_one_line(raw: &str) -> String {
    let flattened = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out: String = flattened.chars().take(200).collect();
    if flattened.chars().count() > 200 {
        out.push('…');
    }
    out
}

// ---------------------------------------------------------------------------
// codex-cloud
// ---------------------------------------------------------------------------

/// The Codex CLI accepts `--limit` values of 1–20 only (20 is also its
/// default), so every page request stays inside that window and larger
/// requests paginate with `--cursor` instead.
const CODEX_PAGE_LIMIT: usize = 20;

/// The process side of `codex cloud list --json`, abstracted so parsing,
/// mapping, and pagination are testable without the Codex CLI installed.
/// `limit` is a per-page cap (at most [`CODEX_PAGE_LIMIT`]); `cursor`
/// continues a previous page's listing.
pub(crate) trait CodexCloudLister: Send + Sync {
    fn list_json(&self, limit: usize, cursor: Option<&str>) -> Result<String>;
}

/// Runs the real `codex` CLI. Its `--json` output is Codex's documented
/// scripting contract, and the CLI handles auth/refresh itself — the same
/// reason the engine shells out to `git` instead of reimplementing it.
struct ExecCodexCli;

impl CodexCloudLister for ExecCodexCli {
    fn list_json(&self, limit: usize, cursor: Option<&str>) -> Result<String> {
        let mut command = std::process::Command::new("codex");
        command.args(["cloud", "list", "--json", "--limit"]);
        command.arg(limit.clamp(1, CODEX_PAGE_LIMIT).to_string());
        if let Some(cursor) = cursor {
            command.arg("--cursor").arg(cursor);
        }
        let output = command
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => anyhow::anyhow!(
                    "the `codex` CLI is not on PATH; install Codex to list cloud tasks"
                ),
                _ => anyhow::Error::from(error).context("could not run `codex cloud list`"),
            })?;
        anyhow::ensure!(
            output.status.success(),
            "`codex cloud list --json` failed ({}): {}",
            output.status,
            excerpt_one_line(&String::from_utf8_lossy(&output.stderr))
        );
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// Map one cloud task from `codex cloud list --json` into a catalog row.
fn map_codex_cloud_task(value: &Value) -> Option<(Candidate, ShallowSession)> {
    let id = string_field(value, "id")?;
    let updated_at = string_field(value, "updated_at");
    let last_activity_ms = updated_at.as_deref().and_then(crate::parse_iso_ms);
    // Status participates in the stamp: an applied or failed task whose
    // timestamp did not move still deserves a re-read.
    let stamp = format!(
        "cloud:{}:{}",
        updated_at.as_deref().unwrap_or("unknown"),
        string_field(value, "status")
            .as_deref()
            .unwrap_or("unknown")
    );
    let session = ShallowSession {
        source: "codex".into(),
        session_id: id.clone(),
        // The task title is Codex's own rendering of the prompt that created
        // the task — the only human-readable identifier the listing offers.
        first_prompt: string_field(value, "title"),
        last_activity_ms,
        raw_path: string_field(value, "url"),
        discovery_state: "shallow".into(),
        ..Default::default()
    };
    let candidate = Candidate {
        source: "codex",
        locator: id.clone(),
        session_id: Some(id),
        recency_hint_ms: last_activity_ms,
        stamp,
    };
    Some((candidate, session))
}

/// One page of `codex cloud list --json` output.
struct CodexCloudPage {
    tasks: Vec<Value>,
    /// Continuation cursor, when the payload carries one.
    cursor: Option<String>,
}

/// Accept both documented shapes: `{"tasks": […], "cursor": …}` and a bare
/// task array (which carries no cursor and therefore ends the walk).
fn parse_codex_cloud_listing(raw: &str) -> Result<CodexCloudPage> {
    let payload: Value =
        serde_json::from_str(raw).context("`codex cloud list --json` output is not JSON")?;
    match &payload {
        Value::Array(tasks) => Ok(CodexCloudPage {
            tasks: tasks.clone(),
            cursor: None,
        }),
        Value::Object(_) => Ok(CodexCloudPage {
            tasks: payload
                .get("tasks")
                .and_then(Value::as_array)
                .cloned()
                .context("`codex cloud list --json` output has no tasks array")?,
            cursor: string_field(&payload, "cursor"),
        }),
        _ => anyhow::bail!("`codex cloud list --json` output has no tasks array"),
    }
}

/// Shallow adapter over the Codex cloud task list.
pub(crate) struct CodexCloudProvider {
    lister: Box<dyn CodexCloudLister>,
    limit: Option<usize>,
    fetched: FetchedRows,
}

impl CodexCloudProvider {
    pub(crate) fn new(lister: Box<dyn CodexCloudLister>, limit: Option<usize>) -> Self {
        Self {
            lister,
            limit,
            fetched: FetchedRows::default(),
        }
    }
}

impl ShallowSessionProvider for CodexCloudProvider {
    fn source(&self) -> &'static str {
        "codex"
    }

    fn location(&self) -> SessionLocation {
        SessionLocation::Remote
    }

    fn enumerate(&self, _env: &DiscoveryEnv<'_>) -> Result<Vec<Candidate>> {
        let mut candidates = Vec::new();
        let mut rows = BTreeMap::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let page_limit = match self.limit {
                Some(limit) => limit
                    .saturating_sub(candidates.len())
                    .clamp(1, CODEX_PAGE_LIMIT),
                None => CODEX_PAGE_LIMIT,
            };
            let raw = self.lister.list_json(page_limit, cursor.as_deref())?;
            let page = parse_codex_cloud_listing(&raw)?;
            for task in &page.tasks {
                if let Some((candidate, session)) = map_codex_cloud_task(task) {
                    rows.insert(candidate.locator.clone(), session);
                    candidates.push(candidate);
                }
            }
            cursor = page.cursor;
            let done = cursor.is_none()
                || page.tasks.is_empty()
                || self.limit.is_some_and(|limit| candidates.len() >= limit);
            if done {
                break;
            }
        }
        *self.fetched.lock().expect("remote connector row cache") = rows;
        Ok(candidates)
    }

    fn read_shallow(
        &self,
        _scan: &ScanEnv<'_>,
        _catalog: Option<&Connection>,
        candidate: &Candidate,
    ) -> Result<Option<ShallowSession>> {
        Ok(take_fetched(&self.fetched, &candidate.locator))
    }
}

#[cfg(test)]
mod tests;
