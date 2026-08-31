//! Targeted, provider-bounded session evidence acquisition.

use super::*;
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use serde::Serialize;
use std::time::Instant;

pub const SESSION_HYDRATION_CONTRACT_VERSION: u32 = 1;
const HYDRATION_PARSER_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct HydrateSessionOptions {
    pub source: String,
    pub session_id: String,
    pub scope: SessionScope,
    pub include_related: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HydrationIndexedThrough {
    pub source_stamp: Option<String>,
    pub last_event_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HydrationEvidence {
    pub prompts: u64,
    pub events: u64,
    pub tool_calls: u64,
    pub related_sessions: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HydrationDiagnostic {
    pub code: String,
    pub message: String,
    pub duration_ms: Option<i64>,
    pub source_bytes: Option<i64>,
    pub records_parsed: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug)]
struct CatalogTarget {
    locator: Option<String>,
    discovery_state: Option<String>,
}

#[derive(Debug)]
struct SourceSnapshot {
    stamp: String,
    bytes: i64,
    records: i64,
    path: Option<PathBuf>,
}

fn hydration_error(code: &str, message: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{code}: {message}")
}

pub fn hydrate_session(options: &HydrateSessionOptions) -> Result<HydrateSessionResult> {
    hydrate_session_at(&default_db_path(), options)
}

pub fn hydrate_session_at(
    db_path: &Path,
    options: &HydrateSessionOptions,
) -> Result<HydrateSessionResult> {
    hydrate_session_at_with_home(db_path, options, &home_dir())
}

fn hydrate_session_at_with_home(
    db_path: &Path,
    options: &HydrateSessionOptions,
    home: &Path,
) -> Result<HydrateSessionResult> {
    validate_options(options)?;
    let started = Instant::now();
    let mut conn = open_db(db_path)?;
    let target = catalog_target(&conn, options)?;
    let snapshot = source_snapshot(options, &target, home)?;
    let previous: Option<(Option<String>, bool)> = conn
        .query_row(
            "SELECT source_stamp, include_related FROM session_hydration_checkpoints \
             WHERE source = ? AND session_id = ? AND location = 'local'",
            params![options.source, options.session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let previous_stamp = previous.as_ref().and_then(|(stamp, _)| stamp.clone());

    if previous_stamp.as_deref() == Some(snapshot.stamp.as_str())
        && (!options.include_related || previous.as_ref().is_some_and(|(_, included)| *included))
        && target.discovery_state.as_deref() == Some("full")
    {
        return build_result(
            &conn,
            options,
            "unchanged",
            snapshot,
            started.elapsed().as_millis() as i64,
        );
    }

    // One selected provider session is one destination transaction. Provider
    // JSONL readers ignore an incomplete final record, and every evidence table
    // has a provider-native uniqueness key, so interruption followed by retry is
    // safe for both new and growing sessions.
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    ingest_selected(&tx, options, &target, snapshot.path.as_deref())?;
    tx.execute(
        "UPDATE sessions SET discovery_state = 'full', source_stamp = ?, parser_version = ? \
         WHERE source = ? AND session_id = ?",
        params![
            snapshot.stamp,
            HYDRATION_PARSER_VERSION,
            options.source,
            options.session_id
        ],
    )?;
    tx.execute(
        "UPDATE session_presences SET discovery_state = 'full', source_stamp = ? \
         WHERE source = ? AND session_id = ? AND location = 'local'",
        params![snapshot.stamp, options.source, options.session_id],
    )?;
    let last_event_at_ms = max_event_time(&tx, &options.source, &options.session_id)?;
    tx.execute(
        "INSERT INTO session_hydration_checkpoints \
         (source, session_id, location, source_stamp, parser_version, last_event_at_ms, source_bytes, records_parsed, include_related, updated_ms) \
         VALUES (?, ?, 'local', ?, ?, ?, ?, ?, ?, ?) \
         ON CONFLICT(source, session_id, location) DO UPDATE SET \
           source_stamp = excluded.source_stamp, parser_version = excluded.parser_version, \
           last_event_at_ms = excluded.last_event_at_ms, source_bytes = excluded.source_bytes, \
           records_parsed = excluded.records_parsed, include_related = excluded.include_related, \
           updated_ms = excluded.updated_ms",
        params![
            options.source,
            options.session_id,
            snapshot.stamp,
            HYDRATION_PARSER_VERSION,
            last_event_at_ms,
            snapshot.bytes,
            snapshot.records,
            options.include_related,
            now_ms(),
        ],
    )?;
    tx.commit()?;

    let status = if previous_stamp.is_some() {
        "updated"
    } else {
        "hydrated"
    };
    build_result(
        &conn,
        options,
        status,
        snapshot,
        started.elapsed().as_millis() as i64,
    )
}

fn validate_options(options: &HydrateSessionOptions) -> Result<()> {
    if options.session_id.trim().is_empty() {
        return Err(hydration_error(
            "INVALID_ARGUMENT",
            "sessionId must not be empty",
        ));
    }
    if !matches!(
        options.source.as_str(),
        "claude" | "codex" | "cursor" | "grok" | "relay" | "opencode"
    ) {
        return Err(hydration_error(
            "INVALID_ARGUMENT",
            format!("unsupported catalog source '{}'", options.source),
        ));
    }
    if options.scope == SessionScope::Remote {
        return Err(hydration_error(
            "HYDRATION_UNSUPPORTED",
            "remote session hydration is not available: no remote provider connector is configured",
        ));
    }
    Ok(())
}

fn catalog_target(conn: &Connection, options: &HydrateSessionOptions) -> Result<CatalogTarget> {
    let row = conn
        .query_row(
            "SELECT p.raw_locator, COALESCE(p.discovery_state, s.discovery_state) \
             FROM sessions s LEFT JOIN session_presences p \
               ON p.source = s.source AND p.session_id = s.session_id AND p.location = 'local' \
             WHERE s.source = ? AND s.session_id = ?",
            params![options.source, options.session_id],
            |row| {
                Ok(CatalogTarget {
                    locator: row.get(0)?,
                    discovery_state: row.get(1)?,
                })
            },
        )
        .optional()?;
    row.ok_or_else(|| {
        hydration_error(
            "SESSION_NOT_FOUND",
            "Run discoverSessions() before hydrating this session.",
        )
    })
}

fn source_snapshot(
    options: &HydrateSessionOptions,
    target: &CatalogTarget,
    home: &Path,
) -> Result<SourceSnapshot> {
    if options.source == "relay" {
        return Err(hydration_error(
            "HYDRATION_UNSUPPORTED",
            "Relay catalog evidence has no configured full-evidence connector",
        ));
    }
    if options.source == "opencode" {
        let path = std::env::var_os("OPENCODE_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/share/opencode/opencode.db"));
        if !path.is_file() {
            return Err(hydration_error(
                "SESSION_SOURCE_UNAVAILABLE",
                format!("OpenCode source {} is unavailable", path.display()),
            ));
        }
        let src = Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
        )?;
        let stamp = src
            .query_row(
                "SELECT printf('%lld:%lld', COALESCE(time_created, 0), \
                    COALESCE(time_updated, time_created, 0)) \
                 FROM session WHERE id = ?",
                [&options.session_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| {
                hydration_error(
                    "SESSION_SOURCE_UNAVAILABLE",
                    format!("OpenCode session '{}' no longer exists", options.session_id),
                )
            })?;
        return Ok(SourceSnapshot {
            stamp,
            bytes: 0,
            // The ingestion query remains session-keyed. Avoid a second count
            // query here so checkpoint resolution never scans the provider's
            // complete `part` table on older stores missing its usual index.
            records: 0,
            path: Some(path),
        });
    }

    let locator = target.locator.as_deref().ok_or_else(|| {
        hydration_error(
            "SESSION_SOURCE_UNAVAILABLE",
            "catalog row has no local provider locator; run discoverSessions() again",
        )
    })?;
    let path = PathBuf::from(locator);
    if !path.is_file() {
        return Err(hydration_error(
            "SESSION_SOURCE_UNAVAILABLE",
            format!(
                "provider source {} disappeared after discovery",
                path.display()
            ),
        ));
    }
    validate_provider_path(&options.source, &path, home)?;
    let mut bytes = path.metadata()?.len() as i64;
    let mut records = complete_jsonl_records(&path)?;
    let mut stamp = if options.source == "grok" {
        grok_session_stamp(&path)?
    } else {
        file_stamp(&path)?
    };
    if options.source == "claude" && options.include_related {
        for sidecar in claude_sidecars(&path, &options.session_id)? {
            stamp.push('|');
            stamp.push_str(&file_stamp(&sidecar)?);
            bytes += sidecar.metadata()?.len() as i64;
            records += complete_jsonl_records(&sidecar)?;
        }
    }
    if options.source == "codex" && options.include_related {
        for child in codex_children(&path, &options.session_id)? {
            stamp.push('|');
            stamp.push_str(&file_stamp(&child)?);
            bytes += child.metadata()?.len() as i64;
            records += complete_jsonl_records(&child)?;
        }
    }
    Ok(SourceSnapshot {
        stamp,
        bytes,
        records,
        path: Some(path),
    })
}

fn validate_provider_path(source: &str, path: &Path, home: &Path) -> Result<()> {
    let roots = match source {
        "claude" => vec![home.join(".claude/projects")],
        "codex" => vec![
            home.join(".codex/sessions"),
            home.join(".codex/archived_sessions"),
        ],
        "cursor" => vec![home.join(".cursor/projects")],
        "grok" => vec![home.join(".grok/sessions")],
        _ => Vec::new(),
    };
    let canonical = fs::canonicalize(path)?;
    let valid = roots
        .iter()
        .filter_map(|root| fs::canonicalize(root).ok())
        .any(|root| canonical.starts_with(root));
    if !valid {
        return Err(hydration_error(
            "SESSION_SOURCE_MISMATCH",
            format!("catalog locator does not belong to the {source} provider root"),
        ));
    }
    Ok(())
}

fn complete_jsonl_records(path: &Path) -> Result<i64> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut records = 0;
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }
        if !line.ends_with('\n') {
            break;
        }
        if serde_json::from_str::<Value>(line.trim_end()).is_ok() {
            records += 1;
        }
    }
    Ok(records)
}

fn ingest_selected(
    conn: &Connection,
    options: &HydrateSessionOptions,
    target: &CatalogTarget,
    path: Option<&Path>,
) -> Result<()> {
    match options.source.as_str() {
        "claude" => ingest_claude(conn, options, path.unwrap()),
        "codex" => ingest_codex(conn, options, path.unwrap()),
        "cursor" => ingest_cursor(conn, options, target, path.unwrap()),
        "grok" => ingest_grok(conn, options, path.unwrap()),
        "opencode" => {
            sync_opencode_session(conn, path.unwrap(), &options.session_id)?;
            Ok(())
        }
        _ => Err(hydration_error(
            "HYDRATION_UNSUPPORTED",
            format!("{} targeted hydration is unavailable", options.source),
        )),
    }
}

fn ingest_claude(conn: &Connection, options: &HydrateSessionOptions, path: &Path) -> Result<()> {
    let meta = scan_claude_session_file(path)?.ok_or_else(|| {
        hydration_error(
            "SESSION_SOURCE_MISMATCH",
            "Claude transcript has no session identity",
        )
    })?;
    if meta.session_id != options.session_id {
        return Err(hydration_error(
            "SESSION_SOURCE_MISMATCH",
            "Claude transcript identity does not match the catalog row",
        ));
    }
    upsert_session(
        conn,
        &meta.session_id,
        "claude",
        meta.cwd.as_deref(),
        meta.git_branch.as_deref(),
        meta.first_ts,
        meta.last_ts,
        meta.last_assistant_text.as_deref(),
        Some(&path.to_string_lossy()),
    )?;
    ingest_claude_transcript(conn, path)?;
    if options.include_related {
        for sidecar in claude_sidecars(path, &options.session_id)? {
            ingest_claude_transcript(conn, &sidecar)?;
        }
    }
    Ok(())
}

fn claude_sidecars(path: &Path, session_id: &str) -> Result<Vec<PathBuf>> {
    let Some(directory) = path.parent() else {
        return Ok(Vec::new());
    };
    Ok(collect_matching_files(directory, "agent-", "jsonl")?
        .into_iter()
        .filter(|candidate| candidate != path)
        .filter(|candidate| {
            scan_claude_session_file(candidate)
                .ok()
                .flatten()
                .is_some_and(|meta| meta.session_id == session_id)
        })
        .collect())
}

fn ingest_codex(conn: &Connection, options: &HydrateSessionOptions, path: &Path) -> Result<()> {
    let meta = read_codex_session_meta(path)?.ok_or_else(|| {
        hydration_error(
            "SESSION_SOURCE_MISMATCH",
            "Codex rollout has no session metadata",
        )
    })?;
    if meta.session_id != options.session_id || meta.is_subagent {
        return Err(hydration_error(
            "SESSION_SOURCE_MISMATCH",
            "Codex rollout identity does not match the selected root session",
        ));
    }
    let outcome = ingest_codex_rollout(conn, path, &meta)?;
    if let Some(first) = outcome.first_ts {
        upsert_session(
            conn,
            &meta.session_id,
            "codex",
            Some(&meta.cwd),
            meta.git_branch.as_deref(),
            first,
            outcome.last_ts.unwrap_or(first),
            outcome.last_assistant_text.as_deref(),
            Some(&path.to_string_lossy()),
        )?;
    }
    if options.include_related {
        ingest_codex_children(conn, options, path)?;
    }
    Ok(())
}

fn ingest_codex_children(
    conn: &Connection,
    options: &HydrateSessionOptions,
    root_path: &Path,
) -> Result<()> {
    for candidate in codex_children(root_path, &options.session_id)? {
        let Some(meta) = read_codex_session_meta(&candidate)? else {
            continue;
        };
        ingest_codex_rollout(conn, &candidate, &meta)?;
        cleanup_codex_subagent_history(conn, &meta.session_id)?;
        cleanup_codex_subagent_registration(conn, &meta.session_id)?;
        conn.execute(
            "INSERT INTO session_relationships \
             (source, parent_session_id, child_session_id, relationship, created_ms) \
             VALUES ('codex', ?, ?, 'delegated', ?) \
             ON CONFLICT(source, parent_session_id, child_session_id) DO NOTHING",
            params![options.session_id, meta.session_id, now_ms()],
        )?;
    }
    Ok(())
}

fn codex_children(root_path: &Path, parent_session_id: &str) -> Result<Vec<PathBuf>> {
    let Some(directory) = root_path.parent() else {
        return Ok(Vec::new());
    };
    Ok(collect_matching_files(directory, "rollout-", "jsonl")?
        .into_iter()
        .filter(|candidate| candidate != root_path)
        .filter(|candidate| {
            read_codex_session_meta(candidate)
                .ok()
                .flatten()
                .is_some_and(|meta| {
                    meta.is_subagent && meta.parent_session_id.as_deref() == Some(parent_session_id)
                })
        })
        .collect())
}

fn ingest_cursor(
    conn: &Connection,
    options: &HydrateSessionOptions,
    _target: &CatalogTarget,
    path: &Path,
) -> Result<()> {
    let project = path
        .ancestors()
        .find(|ancestor| {
            ancestor
                .parent()
                .and_then(Path::file_name)
                .and_then(|s| s.to_str())
                == Some("projects")
        })
        .and_then(Path::file_name)
        .and_then(|s| s.to_str())
        .map(decode_cursor_project);
    let ts_ms = file_modified_ms(path).unwrap_or(0);
    let reader = BufReader::new(fs::File::open(path)?);
    for line in reader.lines().map_while(std::result::Result::ok) {
        if let Some(project) = project.as_deref() {
            ingest_cursor_line(conn, &line, &options.session_id, project, ts_ms)?;
        }
    }
    upsert_session(
        conn,
        &options.session_id,
        "cursor",
        project.as_deref(),
        None,
        ts_ms,
        ts_ms,
        None,
        Some(&path.to_string_lossy()),
    )
}

fn ingest_grok(conn: &Connection, options: &HydrateSessionOptions, path: &Path) -> Result<()> {
    let session = scan_grok_session_file(path)?.ok_or_else(|| {
        hydration_error(
            "SESSION_SOURCE_MISMATCH",
            "Grok source has no session identity",
        )
    })?;
    if session.session_id != options.session_id {
        return Err(hydration_error(
            "SESSION_SOURCE_MISMATCH",
            "Grok source identity does not match the catalog row",
        ));
    }
    ingest_grok_session(conn, &session, &path.to_string_lossy())?;
    Ok(())
}

fn build_result(
    conn: &Connection,
    options: &HydrateSessionOptions,
    status: &str,
    snapshot: SourceSnapshot,
    duration_ms: i64,
) -> Result<HydrateSessionResult> {
    let related_session_ids = if options.include_related {
        related_ids(conn, &options.source, &options.session_id)?
    } else {
        Vec::new()
    };
    let mut ids = vec![options.session_id.clone()];
    ids.extend(related_session_ids.iter().cloned());
    let evidence = evidence_counts(
        conn,
        &options.source,
        &ids,
        related_session_ids.len() as u64,
    )?;
    let last_event_at_ms = ids.iter().try_fold(None, |max, id| {
        let value = max_event_time(conn, &options.source, id)?;
        Ok::<_, anyhow::Error>(match (max, value) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (None, value) | (value, None) => value,
        })
    })?;
    Ok(HydrateSessionResult {
        contract_version: SESSION_HYDRATION_CONTRACT_VERSION,
        source: options.source.clone(),
        session_id: options.session_id.clone(),
        status: status.to_string(),
        discovery_state: "full".to_string(),
        presence: "local".to_string(),
        indexed_through: HydrationIndexedThrough {
            source_stamp: Some(snapshot.stamp),
            last_event_at_ms,
        },
        evidence,
        related_session_ids,
        diagnostics: vec![HydrationDiagnostic {
            code: "HYDRATION_METRICS".to_string(),
            message: "targeted provider evidence acquisition completed".to_string(),
            duration_ms: Some(duration_ms),
            source_bytes: Some(snapshot.bytes),
            records_parsed: Some(snapshot.records),
        }],
    })
}

fn related_ids(conn: &Connection, source: &str, session_id: &str) -> Result<Vec<String>> {
    Ok(conn
        .prepare(
            "SELECT child_session_id FROM session_relationships \
             WHERE source = ? AND parent_session_id = ? ORDER BY child_session_id",
        )?
        .query_map(params![source, session_id], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?)
}

fn evidence_counts(
    conn: &Connection,
    source: &str,
    session_ids: &[String],
    related_sessions: u64,
) -> Result<HydrationEvidence> {
    let mut evidence = HydrationEvidence {
        related_sessions,
        ..Default::default()
    };
    for session_id in session_ids {
        evidence.prompts += count_table(conn, "history", source, session_id)?;
        evidence.events += count_table(conn, "session_events", source, session_id)?;
        evidence.tool_calls += count_table(conn, "tool_calls", source, session_id)?;
    }
    Ok(evidence)
}

fn count_table(conn: &Connection, table: &str, source: &str, session_id: &str) -> Result<u64> {
    // `table` is selected exclusively by the private callers above.
    Ok(conn.query_row(
        &format!("SELECT COUNT(*) FROM {table} WHERE source = ? AND session_id = ?"),
        params![source, session_id],
        |row| row.get::<_, i64>(0),
    )? as u64)
}

fn max_event_time(conn: &Connection, source: &str, session_id: &str) -> Result<Option<i64>> {
    Ok(conn.query_row(
        "SELECT MAX(ts_ms) FROM session_events WHERE source = ? AND session_id = ?",
        params![source, session_id],
        |row| row.get(0),
    )?)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn catalog_row(conn: &Connection, source: &str, session_id: &str, path: Option<&Path>) {
        conn.execute(
            "INSERT INTO sessions (source, session_id, raw_path, discovery_state) \
             VALUES (?, ?, ?, 'shallow')",
            params![
                source,
                session_id,
                path.map(|path| path.to_string_lossy().to_string())
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_presences \
             (source, session_id, location, raw_locator, discovery_state) \
             VALUES (?, ?, 'local', ?, 'shallow')",
            params![
                source,
                session_id,
                path.map(|path| path.to_string_lossy().to_string())
            ],
        )
        .unwrap();
    }

    fn options(source: &str, session_id: &str) -> HydrateSessionOptions {
        HydrateSessionOptions {
            source: source.to_string(),
            session_id: session_id.to_string(),
            scope: SessionScope::Local,
            include_related: true,
        }
    }

    #[test]
    fn catalog_prerequisite_is_stable_and_does_not_discover() {
        let dir = tempfile::tempdir().unwrap();
        let error = hydrate_session_at_with_home(
            &dir.path().join("history.db"),
            &options("claude", "missing"),
            dir.path(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").starts_with("SESSION_NOT_FOUND:"));
        let conn = open_db(&dir.path().join("history.db")).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn claude_hydration_is_incremental_idempotent_and_ignores_partial_tail() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join(".claude/projects/app/session-1.jsonl");
        fs::create_dir_all(transcript.parent().unwrap()).unwrap();
        fs::write(
            &transcript,
            concat!(
                "{\"sessionId\":\"session-1\",\"uuid\":\"u1\",\"cwd\":\"/work/app\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"first prompt\"},\"timestamp\":\"2026-08-31T10:00:00Z\"}\n",
                "{\"sessionId\":\"session-1\",\"uuid\":\"a1\",\"cwd\":\"/work/app\",\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"done\"}]},\"timestamp\":\"2026-08-31T10:00:01Z\"}\n",
            ),
        )
        .unwrap();
        fs::write(
            transcript.parent().unwrap().join("agent-child.jsonl"),
            concat!(
                "{\"sessionId\":\"session-1\",\"uuid\":\"side-u\",\"isSidechain\":true,\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"delegated instruction\"},\"timestamp\":\"2026-08-31T10:00:00Z\"}\n",
                "{\"sessionId\":\"session-1\",\"uuid\":\"side-a\",\"isSidechain\":true,\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"side result\"},\"timestamp\":\"2026-08-31T10:00:01Z\"}\n",
            ),
        )
        .unwrap();
        let db = dir.path().join("history.db");
        let conn = open_db(&db).unwrap();
        catalog_row(&conn, "claude", "session-1", Some(&transcript));
        drop(conn);

        let first =
            hydrate_session_at_with_home(&db, &options("claude", "session-1"), dir.path()).unwrap();
        assert_eq!(first.status, "hydrated");
        assert_eq!(first.evidence.prompts, 1);
        assert_eq!(first.evidence.events, 3);

        let second =
            hydrate_session_at_with_home(&db, &options("claude", "session-1"), dir.path()).unwrap();
        assert_eq!(second.status, "unchanged");
        assert_eq!(second.evidence.events, 3);

        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&transcript)
            .unwrap();
        write!(file, "{{\"sessionId\":\"session-1\",\"uuid\":\"u2\"").unwrap();
        drop(file);
        let partial =
            hydrate_session_at_with_home(&db, &options("claude", "session-1"), dir.path()).unwrap();
        assert_eq!(partial.status, "updated");
        assert_eq!(partial.evidence.events, 3);

        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&transcript)
            .unwrap();
        writeln!(file, ",\"cwd\":\"/work/app\",\"type\":\"user\",\"message\":{{\"role\":\"user\",\"content\":\"second prompt\"}},\"timestamp\":\"2026-08-31T10:00:02Z\"}}").unwrap();
        drop(file);
        let appended =
            hydrate_session_at_with_home(&db, &options("claude", "session-1"), dir.path()).unwrap();
        assert_eq!(appended.status, "updated");
        assert_eq!(appended.evidence.prompts, 2);
        assert_eq!(appended.evidence.events, 4);
    }

    #[test]
    fn identity_mismatch_rolls_back_catalog_upgrade() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join(".claude/projects/app/wrong.jsonl");
        fs::create_dir_all(transcript.parent().unwrap()).unwrap();
        fs::write(
            &transcript,
            "{\"sessionId\":\"different\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"nope\"}}\n",
        )
        .unwrap();
        let db = dir.path().join("history.db");
        let conn = open_db(&db).unwrap();
        catalog_row(&conn, "claude", "selected", Some(&transcript));
        drop(conn);
        let error = hydrate_session_at_with_home(&db, &options("claude", "selected"), dir.path())
            .unwrap_err();
        assert!(format!("{error:#}").starts_with("SESSION_SOURCE_MISMATCH:"));
        let conn = open_db(&db).unwrap();
        let state: String = conn
            .query_row(
                "SELECT discovery_state FROM sessions WHERE source='claude' AND session_id='selected'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "shallow");
        let checkpoints: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_hydration_checkpoints",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(checkpoints, 0);
    }

    #[test]
    fn opencode_hydration_queries_only_the_selected_session() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join(".local/share/opencode/opencode.db");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        let src = Connection::open(&source).unwrap();
        src.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT, time_created INTEGER, time_updated INTEGER); \
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT); \
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT); \
             INSERT INTO session VALUES ('selected', '/work/selected', 1, 2), ('unrelated', '/work/other', 3, 4); \
             INSERT INTO message VALUES ('m1', 'selected', 1, '{\"role\":\"user\"}'), ('m2', 'unrelated', 3, '{\"role\":\"user\"}'); \
             INSERT INTO part VALUES ('p1', 'm1', 'selected', 2, '{\"type\":\"text\",\"text\":\"selected prompt\"}'), ('p2', 'm2', 'unrelated', 4, '{\"type\":\"text\",\"text\":\"must not ingest\"}');",
        )
        .unwrap();
        drop(src);
        let db = dir.path().join("history.db");
        let conn = open_db(&db).unwrap();
        catalog_row(&conn, "opencode", "selected", None);
        drop(conn);
        let result =
            hydrate_session_at_with_home(&db, &options("opencode", "selected"), dir.path())
                .unwrap();
        assert_eq!(result.evidence.prompts, 1);
        let conn = open_db(&db).unwrap();
        let unrelated: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM history WHERE source='opencode' AND session_id='unrelated'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(unrelated, 0);
    }

    #[test]
    fn codex_related_threads_keep_identity_and_never_become_human_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let day = dir.path().join(".codex/sessions/2026/08/31");
        fs::create_dir_all(&day).unwrap();
        let root = day.join("rollout-root.jsonl");
        let child = day.join("rollout-child.jsonl");
        fs::write(
            &root,
            concat!(
                "{\"timestamp\":\"2026-08-31T10:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"root\",\"cwd\":\"/work/app\"}}\n",
                "{\"timestamp\":\"2026-08-31T10:00:01Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"root prompt\"}}\n",
                "{\"timestamp\":\"2026-08-31T10:00:02Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"agent_message\",\"message\":\"root answer\"}}\n",
            ),
        )
        .unwrap();
        fs::write(
            &child,
            concat!(
                "{\"timestamp\":\"2026-08-31T10:00:03Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"child\",\"session_id\":\"root\",\"cwd\":\"/work/app\",\"thread_source\":\"subagent\"}}\n",
                "{\"timestamp\":\"2026-08-31T10:00:04Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"delegated task\"}}\n",
                "{\"timestamp\":\"2026-08-31T10:00:05Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"agent_message\",\"message\":\"child answer\"}}\n",
            ),
        )
        .unwrap();
        let db = dir.path().join("history.db");
        let conn = open_db(&db).unwrap();
        catalog_row(&conn, "codex", "root", Some(&root));
        drop(conn);

        let mut request = options("codex", "root");
        request.include_related = false;
        let root_only = hydrate_session_at_with_home(&db, &request, dir.path()).unwrap();
        assert!(root_only.related_session_ids.is_empty());

        request.include_related = true;
        let with_child = hydrate_session_at_with_home(&db, &request, dir.path()).unwrap();
        assert_eq!(with_child.status, "updated");
        assert_eq!(with_child.related_session_ids, vec!["child"]);
        let conn = open_db(&db).unwrap();
        let child_events: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM session_events WHERE source='codex' AND session_id='child'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(child_events >= 2);
        let child_prompts: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM history WHERE source='codex' AND session_id='child'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(child_prompts, 0);
        let child_catalog: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE source='codex' AND session_id='child'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(child_catalog, 0);
    }

    #[test]
    fn thousands_of_unrelated_catalog_rows_do_not_expand_hydration_work() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join(".claude/projects/app/selected.jsonl");
        fs::create_dir_all(transcript.parent().unwrap()).unwrap();
        fs::write(
            &transcript,
            "{\"sessionId\":\"selected\",\"uuid\":\"u1\",\"cwd\":\"/work/app\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"bounded\"},\"timestamp\":1}\n",
        )
        .unwrap();
        let db = dir.path().join("history.db");
        let conn = open_db(&db).unwrap();
        conn.execute_batch(
            "WITH RECURSIVE seq(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM seq WHERE x < 5000) \
             INSERT INTO sessions (source, session_id, discovery_state) \
             SELECT 'codex', 'unrelated-' || x, 'shallow' FROM seq;",
        )
        .unwrap();
        catalog_row(&conn, "claude", "selected", Some(&transcript));
        drop(conn);
        let result =
            hydrate_session_at_with_home(&db, &options("claude", "selected"), dir.path()).unwrap();
        assert_eq!(result.evidence.prompts, 1);
        assert_eq!(result.diagnostics[0].records_parsed, Some(1));
        let conn = open_db(&db).unwrap();
        let full: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE discovery_state='full'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(full, 1);
    }

    #[test]
    fn simultaneous_hydration_remains_deduplicated() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join(".claude/projects/app/concurrent.jsonl");
        fs::create_dir_all(transcript.parent().unwrap()).unwrap();
        fs::write(
            &transcript,
            "{\"sessionId\":\"concurrent\",\"uuid\":\"u1\",\"cwd\":\"/work/app\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"once\"},\"timestamp\":1}\n",
        )
        .unwrap();
        let db = dir.path().join("history.db");
        let conn = open_db(&db).unwrap();
        catalog_row(&conn, "claude", "concurrent", Some(&transcript));
        drop(conn);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles = (0..2)
            .map(|_| {
                let db = db.clone();
                let home = dir.path().to_path_buf();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    hydrate_session_at_with_home(&db, &options("claude", "concurrent"), &home)
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
        let conn = open_db(&db).unwrap();
        let counts: (i64, i64) = conn
            .query_row(
                "SELECT \
                   (SELECT COUNT(*) FROM history WHERE source='claude' AND session_id='concurrent'), \
                   (SELECT COUNT(*) FROM session_events WHERE source='claude' AND session_id='concurrent')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1));
    }
}
