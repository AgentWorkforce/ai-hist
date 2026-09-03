//! WS-9 cloud-sync increment 2a: the outbox builder (pure sync logic, no network).
//!
//! Reads new local rows past a resume cursor, maps them to WS-1 convergence envelopes
//! (via [`crate::convergence`]), applies the incognito exclusion, and returns the batch
//! plus the advanced cursor. Network I/O (POST `/v1/ingest`, `rth_` auth) lives in the
//! binding layer (the `ai-hist` binary) per the no-async-in-core rule — this module only
//! does sync rusqlite reads, so it is fully unit-testable without a server.
//!
//! Cursor model (mirrors burn's `archive_state` watermark): monotonic `history.id`,
//! `trajectories.rowid` plus a `trajectories.updated_ms` watermark so rows revised after
//! first sync re-push (the server upsert is already safe), and `session_commit_links.id`
//! for `session_outcome` envelopes.

use crate::convergence::{
    map_history_entry_with, map_session_outcome, map_trajectory, normalize_home_path,
    resolve_project_id, ConvergenceEnvelope, SessionCommitLink, TrajectoryRow, UNKNOWN_PROJECT,
};
use crate::{
    session_file_edits, session_file_edits_page, HistoryEntry, SessionEvidenceCursor,
    SessionFileEdit,
};
use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Resume watermarks for incremental cloud sync (the local cursor store). Persisted by the
/// binding layer (single cursor store) and advanced to the server-confirmed values after a
/// successful `/v1/ingest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SyncCursor {
    /// Highest `history.id` included in a synced batch.
    #[serde(default)]
    pub history_id: i64,
    /// Highest `trajectories.rowid` included in a synced batch.
    #[serde(default)]
    pub trajectory_rowid: i64,
    /// Highest `trajectories.updated_ms` included in a synced batch. Catches rows that
    /// were revised after their rowid first crossed the cursor.
    #[serde(default)]
    pub trajectory_updated_ms: i64,
    /// Highest `session_commit_links.id` included in a synced batch.
    #[serde(default)]
    pub commit_link_id: i64,
}

impl SyncCursor {
    /// Advance each watermark independently so a stale writer cannot rewind one
    /// by publishing an older value of another.
    pub fn merge_max(&self, other: &Self) -> Self {
        Self {
            history_id: self.history_id.max(other.history_id),
            trajectory_rowid: self.trajectory_rowid.max(other.trajectory_rowid),
            trajectory_updated_ms: self.trajectory_updated_ms.max(other.trajectory_updated_ms),
            commit_link_id: self.commit_link_id.max(other.commit_link_id),
        }
    }
}

/// The next outbox batch: the envelopes to POST and the cursor they advance to.
#[derive(Debug, Clone, PartialEq)]
pub struct OutboxBatch {
    pub records: Vec<ConvergenceEnvelope>,
    pub cursor: SyncCursor,
}

/// Build the next batch of convergence envelopes from local rows past `cursor`.
///
/// - `limit` caps rows scanned **per source** (history, trajectories).
/// - `incognito` holds session ids (history `session_id` / trajectory `id`) to exclude —
///   incognito rows are skipped but still advance the cursor so they are never re-scanned.
/// - The returned cursor advances to the max id/rowid *scanned* (not just emitted), so
///   skipped/empty rows don't cause re-scanning on the next call.
pub fn build_outbox_batch(
    conn: &Connection,
    cursor: &SyncCursor,
    limit: usize,
    incognito: &HashSet<String>,
) -> Result<OutboxBatch> {
    let limit = limit.max(1) as i64;
    // The server caps an ingest batch at 1000 records. A single trajectory row — especially
    // a compacted roll-up — expands to many convergence events (decisions + findings +
    // reflections), so the batch must be bounded on EMITTED records, not rows scanned, or a
    // handful of roll-ups blows past the cap and the push 400s.
    const MAX_RECORDS: usize = 900;
    let mut records = Vec::new();
    let mut next = cursor.clone();
    let mut remotes: HashMap<String, Option<String>> = HashMap::new();
    let mut file_cache: HashMap<(String, String), Vec<String>> = HashMap::new();

    // --- history (prompts) — append-only, watermark on id ---
    {
        let mut stmt = conn.prepare(
            "SELECT id, source, session_id, project, prompt, prompt_hash, timestamp_ms \
             FROM history WHERE id > ?1 ORDER BY id ASC LIMIT ?2",
        )?;
        let rows = stmt.query_map([cursor.history_id, limit], |r| {
            Ok(HistoryEntry {
                id: r.get(0)?,
                source: r.get(1)?,
                session_id: r.get(2)?,
                project: r.get(3)?,
                prompt: r.get(4)?,
                prompt_hash: r.get(5)?,
                timestamp_ms: r.get(6)?,
            })
        })?;
        for row in rows {
            let entry = row?;
            // Stop before consuming this row if the batch is full, so the cursor does not
            // advance past an un-emitted row (the next batch resumes from here).
            if records.len() >= MAX_RECORDS {
                break;
            }
            next.history_id = next.history_id.max(entry.id);
            // incognito: skip rows whose session is suppressed (still advances cursor)
            if let Some(sid) = &entry.session_id {
                if incognito.contains(sid) {
                    continue;
                }
            }
            let git_remote = git_remote_for_entry(conn, &entry, &mut remotes);
            let files = match &entry.session_id {
                Some(sid) => session_files(conn, &mut file_cache, Some(&entry.source), sid)?,
                None => Vec::new(),
            };
            records.push(map_history_entry_with(&entry, git_remote.as_deref(), files));
        }
    }

    // --- trajectories (decisions/retro) — rowid + updated_ms watermarks ---
    let updated_watermark = trajectory_updated_watermark(conn, cursor)?;
    next.trajectory_updated_ms = next.trajectory_updated_ms.max(updated_watermark);
    {
        let mut stmt = conn.prepare(
            "SELECT rowid, id, persona_id, project_id, task_title, task_description, status, \
             decisions_json, retrospective_json, timestamp_ms, updated_ms, path \
             FROM trajectories WHERE rowid > ?1 OR updated_ms > ?2 \
             ORDER BY updated_ms ASC, rowid ASC LIMIT ?3",
        )?;
        let raw = stmt.query_map(
            rusqlite::params![cursor.trajectory_rowid, updated_watermark, limit],
            |r| {
                Ok(TrajRowOwned {
                    rowid: r.get(0)?,
                    id: r.get(1)?,
                    persona_id: r.get(2)?,
                    project_id: r.get(3)?,
                    task_title: r.get(4)?,
                    task_description: r.get(5)?,
                    status: r.get(6)?,
                    decisions_json: r.get(7)?,
                    retrospective_json: r.get(8)?,
                    timestamp_ms: r.get(9)?,
                    updated_ms: r.get(10)?,
                    path: r.get(11)?,
                })
            },
        )?;
        for row in raw {
            let t = row?;
            if incognito.contains(&t.id) {
                next.trajectory_rowid = next.trajectory_rowid.max(t.rowid);
                next.trajectory_updated_ms = next.trajectory_updated_ms.max(t.updated_ms);
                continue;
            }
            let mut mapped = map_trajectory(&TrajectoryRow {
                id: &t.id,
                persona_id: t.persona_id.as_deref(),
                project_id: t.project_id.as_deref(),
                task_title: t.task_title.as_deref(),
                task_description: t.task_description.as_deref(),
                status: t.status.as_deref(),
                task_ref: None, // not in local store yet (see convergence::TrajectoryRow)
                decisions_json: &t.decisions_json,
                retrospective_json: &t.retrospective_json,
                timestamp_ms: t.timestamp_ms,
            });
            // A compacted roll-up expands to many events. If adding this row would exceed the
            // batch cap and we already have records, defer it to the next batch — leave the
            // cursor before it so nothing is skipped. (A lone row under the cap always fits.)
            if !records.is_empty() && records.len() + mapped.len() > MAX_RECORDS {
                break;
            }
            let files = trajectory_files(conn, &mut file_cache, &t.id, t.path.as_deref())?;
            for env in &mut mapped {
                env.files_touched = files.clone();
            }
            next.trajectory_rowid = next.trajectory_rowid.max(t.rowid);
            next.trajectory_updated_ms = next.trajectory_updated_ms.max(t.updated_ms);
            records.extend(mapped);
        }
    }

    // --- session_commit_links → kind=session_outcome (one envelope per link row) ---
    {
        let mut stmt = conn.prepare(
            "SELECT id, source, session_id, repo, branch, commit_sha, match_method, confidence, \
             files_json, numstat_json, evidence_json, created_at_ms \
             FROM session_commit_links WHERE id > ?1 ORDER BY id ASC LIMIT ?2",
        )?;
        let rows = stmt.query_map([cursor.commit_link_id, limit], |r| {
            Ok(CommitLinkOwned {
                id: r.get(0)?,
                source: r.get(1)?,
                session_id: r.get(2)?,
                repo: r.get(3)?,
                branch: r.get(4)?,
                commit_sha: r.get(5)?,
                match_method: r.get(6)?,
                confidence: r.get(7)?,
                files_json: r.get(8)?,
                numstat_json: r.get(9)?,
                evidence_json: r.get(10)?,
                created_at_ms: r.get(11)?,
            })
        })?;
        for row in rows {
            let link = row?;
            if records.len() >= MAX_RECORDS {
                break;
            }
            next.commit_link_id = next.commit_link_id.max(link.id);
            if incognito.contains(&link.session_id) {
                continue;
            }
            let files = session_files(conn, &mut file_cache, Some(&link.source), &link.session_id)?;
            let project_id = session_project_id(
                conn,
                &link.source,
                &link.session_id,
                link.repo.as_deref(),
                &mut remotes,
            );
            records.push(map_session_outcome(
                &SessionCommitLink {
                    source: &link.source,
                    session_id: &link.session_id,
                    repo: link.repo.as_deref(),
                    branch: link.branch.as_deref(),
                    commit_sha: &link.commit_sha,
                    match_method: &link.match_method,
                    confidence: link.confidence,
                    files_json: link.files_json.as_deref(),
                    numstat_json: link.numstat_json.as_deref(),
                    evidence_json: link.evidence_json.as_deref(),
                    created_at_ms: link.created_at_ms,
                },
                &project_id,
                files,
            ));
        }
    }

    Ok(OutboxBatch {
        records,
        cursor: next,
    })
}

/// Owned trajectory row (rusqlite can't borrow across the row closure).
struct TrajRowOwned {
    rowid: i64,
    id: String,
    persona_id: Option<String>,
    project_id: Option<String>,
    task_title: Option<String>,
    task_description: Option<String>,
    status: Option<String>,
    decisions_json: String,
    retrospective_json: String,
    timestamp_ms: i64,
    updated_ms: i64,
    path: Option<String>,
}

struct CommitLinkOwned {
    id: i64,
    source: String,
    session_id: String,
    repo: Option<String>,
    branch: Option<String>,
    commit_sha: String,
    match_method: String,
    confidence: f64,
    files_json: Option<String>,
    numstat_json: Option<String>,
    evidence_json: Option<String>,
    created_at_ms: i64,
}

/// Old cursors (pre-watermark) have `trajectory_updated_ms = 0`. Seed from already-synced
/// rows so the first push after upgrade does not replay every trajectory.
fn trajectory_updated_watermark(conn: &Connection, cursor: &SyncCursor) -> Result<i64> {
    if cursor.trajectory_updated_ms > 0 || cursor.trajectory_rowid <= 0 {
        return Ok(cursor.trajectory_updated_ms);
    }
    Ok(conn.query_row(
        "SELECT COALESCE(MAX(updated_ms), 0) FROM trajectories WHERE rowid <= ?1",
        [cursor.trajectory_rowid],
        |r| r.get(0),
    )?)
}

fn git_remote_for_entry(
    conn: &Connection,
    entry: &HistoryEntry,
    remotes: &mut HashMap<String, Option<String>>,
) -> Option<String> {
    if entry
        .project
        .as_deref()
        .map(str::trim)
        .is_some_and(|s| !s.is_empty())
    {
        return None;
    }
    let sid = entry.session_id.as_deref()?;
    let cwd = session_cwd(conn, &entry.source, sid)?;
    remotes
        .entry(cwd.clone())
        .or_insert_with(|| git_origin_url(&cwd))
        .clone()
}

fn session_cwd(conn: &Connection, source: &str, session_id: &str) -> Option<String> {
    conn.query_row(
        "SELECT cwd FROM sessions WHERE source = ?1 AND session_id = ?2",
        rusqlite::params![source, session_id],
        |r| r.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
    .map(|s| s.trim().to_string())
    .filter(|s| !s.is_empty())
}

fn git_origin_url(cwd: &str) -> Option<String> {
    if !Path::new(cwd).is_dir() {
        return None;
    }
    let out = std::process::Command::new("git")
        .args(["-C", cwd, "remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!url.is_empty()).then_some(url)
}

fn session_project_id(
    conn: &Connection,
    source: &str,
    session_id: &str,
    repo: Option<&str>,
    remotes: &mut HashMap<String, Option<String>>,
) -> String {
    let project: Option<String> = conn
        .query_row(
            "SELECT project FROM history WHERE source = ?1 AND session_id = ?2 \
             AND project IS NOT NULL AND trim(project) != '' LIMIT 1",
            rusqlite::params![source, session_id],
            |r| r.get(0),
        )
        .ok();
    let remote = session_cwd(conn, source, session_id).and_then(|cwd| {
        remotes
            .entry(cwd.clone())
            .or_insert_with(|| git_origin_url(&cwd))
            .clone()
    });
    let resolved = resolve_project_id(project.as_deref(), remote.as_deref());
    if resolved != UNKNOWN_PROJECT {
        resolved
    } else {
        resolve_project_id(repo, None)
    }
}

fn session_files(
    conn: &Connection,
    cache: &mut HashMap<(String, String), Vec<String>>,
    source: Option<&str>,
    session_id: &str,
) -> Result<Vec<String>> {
    let key = (source.unwrap_or("*").to_string(), session_id.to_string());
    if let Some(hit) = cache.get(&key) {
        return Ok(hit.clone());
    }
    let edits = match source {
        Some(source) => collect_session_file_edits(conn, source, session_id)?,
        None => session_file_edits(conn, session_id, None)?,
    };
    let mut deduped = Vec::new();
    for edit in edits {
        let trimmed = edit.file_path.trim();
        if !trimmed.is_empty() && !deduped.iter().any(|existing| existing == trimmed) {
            deduped.push(trimmed.to_string());
        }
    }
    cache.insert(key, deduped.clone());
    Ok(deduped)
}

/// Page through [`session_file_edits_page`] until the session is exhausted.
fn collect_session_file_edits(
    conn: &Connection,
    source: &str,
    session_id: &str,
) -> Result<Vec<SessionFileEdit>> {
    let mut all = Vec::new();
    let mut after: Option<SessionEvidenceCursor> = None;
    loop {
        let page = session_file_edits_page(conn, source, session_id, 1_000, after.as_ref())?;
        let next = page.next_cursor;
        all.extend(page.file_edits);
        match next {
            Some(cursor) => after = Some(cursor),
            None => break,
        }
    }
    Ok(all)
}

fn trajectory_files(
    conn: &Connection,
    cache: &mut HashMap<(String, String), Vec<String>>,
    trajectory_id: &str,
    path: Option<&str>,
) -> Result<Vec<String>> {
    let mut files = session_files(conn, cache, None, trajectory_id)?;
    if let Some(origin) = path.and_then(learn_origin_session) {
        for file in session_files(conn, cache, None, origin)? {
            if !files.iter().any(|existing| existing == &file) {
                files.push(file);
            }
        }
    }
    Ok(files
        .into_iter()
        .map(|f| normalize_home_path(f.trim()))
        .filter(|f| !f.is_empty())
        .collect())
}

fn learn_origin_session(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("learn://")?;
    rest.split_once('/')
        .map(|(_, session)| session)
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_db, insert_history};

    fn mem() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        conn
    }

    fn add_history(conn: &Connection, session: &str, prompt: &str, ts: i64) {
        add_history_project(conn, session, None, prompt, ts);
    }

    fn add_history_project(
        conn: &Connection,
        session: &str,
        project: Option<&str>,
        prompt: &str,
        ts: i64,
    ) {
        insert_history(
            conn,
            &HistoryEntry {
                id: 0,
                source: "claude".into(),
                session_id: Some(session.into()),
                project: project.map(str::to_string),
                prompt: prompt.into(),
                prompt_hash: Some(crate::prompt_hash(prompt)),
                timestamp_ms: ts,
            },
        )
        .unwrap();
    }

    fn add_trajectory(conn: &Connection, id: &str, decisions: &str, retro: &str) {
        add_trajectory_at(conn, id, decisions, retro, 1, None);
    }

    fn add_trajectory_at(
        conn: &Connection,
        id: &str,
        decisions: &str,
        retro: &str,
        updated_ms: i64,
        path: Option<&str>,
    ) {
        conn.execute(
            "INSERT INTO trajectories (id, version, persona_id, project_id, task_title, \
             task_description, status, decisions_json, retrospective_json, search_text, path, \
             updated_ms, timestamp_ms) VALUES (?,1,?,?,?,?,?,?,?,?,?,?,?)",
            rusqlite::params![
                id,
                "planner",
                "proj",
                "Build forms",
                "desc",
                "completed",
                decisions,
                retro,
                "search",
                path,
                updated_ms,
                1_782_036_000_000i64
            ],
        )
        .unwrap();
    }

    fn add_file_edit(conn: &Connection, session: &str, path: &str, tool: &str) {
        conn.execute(
            "INSERT INTO file_edits (source, session_id, tool_use_id, file_path, tool_name) \
             VALUES ('claude', ?, ?, ?, ?)",
            rusqlite::params![session, format!("tool_{path}"), path, tool],
        )
        .unwrap();
    }

    fn add_commit_link(conn: &Connection, session: &str, sha: &str, method: &str) {
        add_commit_link_evidence(conn, session, sha, method, None);
    }

    fn add_commit_link_evidence(
        conn: &Connection,
        session: &str,
        sha: &str,
        method: &str,
        evidence_json: Option<&str>,
    ) {
        conn.execute(
            "INSERT INTO session_commit_links \
             (source, session_id, repo, branch, commit_sha, match_method, confidence, \
              files_json, numstat_json, evidence_json, created_at_ms) \
             VALUES ('claude', ?, '/Users/khaliqgant/Projects/relayhistory', 'main', ?, ?, 0.9, \
                     ?, ?, ?, 1_782_036_000_000)",
            rusqlite::params![
                session,
                sha,
                method,
                r#"["src/lib.rs"]"#,
                r#"[{"path":"src/lib.rs","additions":2,"deletions":0}]"#,
                evidence_json,
            ],
        )
        .unwrap();
    }

    fn init_git_repo(origin: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let status = std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(status.status.success(), "git init failed");
        let add = std::process::Command::new("git")
            .args(["remote", "add", "origin", origin])
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(add.status.success(), "git remote add failed");
        dir
    }

    #[test]
    fn builds_batch_and_advances_cursor() {
        let conn = mem();
        add_history(&conn, "s1", "first prompt", 1);
        add_history(&conn, "s1", "second prompt", 2);
        let none = HashSet::new();

        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        assert_eq!(batch.records.len(), 2);
        assert_eq!(batch.cursor.history_id, 2);
        assert!(batch.records.iter().all(|r| r.kind == "prompt"));

        // a second call from the advanced cursor yields nothing new
        let empty = build_outbox_batch(&conn, &batch.cursor, 100, &none).unwrap();
        assert!(empty.records.is_empty());
        assert_eq!(empty.cursor, batch.cursor);
    }

    #[test]
    fn incognito_sessions_are_excluded_but_advance_cursor() {
        let conn = mem();
        add_history(&conn, "public", "keep me", 1);
        add_history(&conn, "secret", "drop me", 2);
        let incognito: HashSet<String> = ["secret".to_string()].into_iter().collect();

        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &incognito).unwrap();
        assert_eq!(batch.records.len(), 1);
        assert_eq!(batch.records[0].content, "keep me");
        // cursor still advances past the skipped incognito row (id 2) — never re-scanned
        assert_eq!(batch.cursor.history_id, 2);
    }

    #[test]
    fn trajectories_fan_out_into_batch() {
        let conn = mem();
        add_trajectory(
            &conn,
            "traj-1",
            r#"[{"chosen":"Formik"}]"#,
            r#"{"summary":"shipped","learnings":["L0"],"confidence":0.8}"#,
        );
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        // decision + summary + learning = 3 events, all trajectory lens
        assert_eq!(batch.records.len(), 3);
        assert!(batch
            .records
            .iter()
            .all(|r| r.lens.as_deref() == Some("trajectories")));
        assert!(batch
            .records
            .iter()
            .any(|r| r.event_id == "decision:traj-1:0"));
        assert!(batch
            .records
            .iter()
            .any(|r| r.event_id == "finding:traj-1:learning:0"));
        assert_eq!(batch.cursor.trajectory_rowid, 1);
    }

    #[test]
    fn incognito_excludes_trajectory_by_id() {
        let conn = mem();
        add_trajectory(&conn, "secret-traj", "[]", r#"{"summary":"hidden"}"#);
        let incognito: HashSet<String> = ["secret-traj".to_string()].into_iter().collect();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &incognito).unwrap();
        assert!(batch.records.is_empty());
        assert_eq!(batch.cursor.trajectory_rowid, 1); // still advanced
    }

    #[test]
    fn limit_caps_rows_per_source() {
        let conn = mem();
        for i in 1..=5 {
            add_history(&conn, "s", &format!("p{i}"), i);
        }
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 2, &none).unwrap();
        assert_eq!(batch.records.len(), 2);
        assert_eq!(batch.cursor.history_id, 2);
    }

    #[test]
    fn cursor_round_trips_through_json() {
        let c = SyncCursor {
            history_id: 7,
            trajectory_rowid: 3,
            trajectory_updated_ms: 99,
            commit_link_id: 4,
        };
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<SyncCursor>(&s).unwrap(), c);
        let old: SyncCursor =
            serde_json::from_str(r#"{"history_id":7,"trajectory_rowid":3}"#).unwrap();
        assert_eq!(old.trajectory_updated_ms, 0);
        assert_eq!(old.commit_link_id, 0);
    }

    #[test]
    fn outbox_batch_project_id_is_never_null() {
        let conn = mem();
        add_history_project(
            &conn,
            "s-path",
            Some("/Users/khaliqgant/Projects/relayhistory"),
            "path project prompt",
            1,
        );
        add_history(&conn, "s-unknown", "no project", 2);
        add_trajectory(
            &conn,
            "traj-1",
            r#"[{"chosen":"Formik"}]"#,
            r#"{"summary":"shipped"}"#,
        );
        add_commit_link(&conn, "s-path", "abc111", "cwd");
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        assert!(!batch.records.is_empty());
        assert!(batch.records.iter().all(|r| r.project_id.is_some()));
        assert!(batch
            .records
            .iter()
            .all(|r| r.project_id.as_deref() != Some("")));
        let path_prompt = batch
            .records
            .iter()
            .find(|r| r.kind == "prompt" && r.session_id == "s-path")
            .unwrap();
        assert_eq!(path_prompt.project_id.as_deref(), Some("relayhistory"));
        let unknown_prompt = batch
            .records
            .iter()
            .find(|r| r.kind == "prompt" && r.session_id == "s-unknown")
            .unwrap();
        assert_eq!(unknown_prompt.project_id.as_deref(), Some(UNKNOWN_PROJECT));
        assert!(batch.records.iter().any(|r| r.kind == "session_outcome"));
    }

    #[test]
    fn git_remote_of_session_cwd_becomes_repo_slug() {
        let repo = init_git_repo("git@github.com:AgentWorkforce/relayhistory.git");
        let conn = mem();
        conn.execute(
            "INSERT INTO sessions (session_id, source, cwd, parser_version) VALUES (?, 'claude', ?, 1)",
            rusqlite::params!["s-git", repo.path().display().to_string()],
        )
        .unwrap();
        add_history(&conn, "s-git", "work in the cloned repo", 1);
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        assert_eq!(batch.records.len(), 1);
        assert_eq!(batch.records[0].kind, "prompt");
        assert_eq!(
            batch.records[0].project_id.as_deref(),
            Some("AgentWorkforce/relayhistory")
        );
        let v = serde_json::to_value(&batch.records[0]).unwrap();
        assert_eq!(v["projectId"], "AgentWorkforce/relayhistory");
        assert!(!v["projectId"].is_null());
    }

    #[test]
    fn files_touched_on_session_and_trajectory_envelopes() {
        let conn = mem();
        add_history_project(
            &conn,
            "s1",
            Some("/Users/me/Projects/relayhistory"),
            "edit the mapper",
            1,
        );
        add_file_edit(
            &conn,
            "s1",
            "/Users/me/Projects/relayhistory/src/outbox.rs",
            "Edit",
        );
        add_file_edit(
            &conn,
            "s1",
            "crates/ai-hist-core/src/convergence.rs",
            "Write",
        );
        add_trajectory_at(
            &conn,
            "learn_abc",
            r#"[{"chosen":"slug the project"}]"#,
            r#"{"summary":"shipped"}"#,
            5,
            Some("learn://claude/s1"),
        );
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        let prompts: Vec<_> = batch
            .records
            .iter()
            .filter(|r| r.kind == "prompt")
            .collect();
        assert_eq!(prompts.len(), 1);
        assert!(prompts[0]
            .files_touched
            .iter()
            .any(|f| f.ends_with("src/outbox.rs")));
        assert!(prompts[0]
            .files_touched
            .iter()
            .any(|f| f.ends_with("crates/ai-hist-core/src/convergence.rs")));
        let traj: Vec<_> = batch
            .records
            .iter()
            .filter(|r| r.lens.as_deref() == Some("trajectories"))
            .collect();
        assert!(!traj.is_empty());
        assert!(traj
            .iter()
            .all(|r| r.files_touched == prompts[0].files_touched));
        let v = serde_json::to_value(prompts[0]).unwrap();
        assert!(v["filesTouched"].as_array().unwrap().len() >= 2);
    }

    #[test]
    fn revised_trajectory_is_re_pushed_via_updated_ms_watermark() {
        let conn = mem();
        add_trajectory(
            &conn,
            "traj-1",
            r#"[{"chosen":"Formik"}]"#,
            r#"{"summary":"first draft"}"#,
        );
        let none = HashSet::new();
        let first = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        assert!(first
            .records
            .iter()
            .any(|r| r.content.contains("first draft")));
        assert!(first.cursor.trajectory_rowid >= 1);
        assert!(first.cursor.trajectory_updated_ms >= 1);

        let empty = build_outbox_batch(&conn, &first.cursor, 100, &none).unwrap();
        assert!(empty
            .records
            .iter()
            .all(|r| r.lens.as_deref() != Some("trajectories")));

        conn.execute(
            "UPDATE trajectories SET retrospective_json = ?, updated_ms = ? WHERE id = ?",
            rusqlite::params![r#"{"summary":"amended after the run"}"#, 50i64, "traj-1"],
        )
        .unwrap();
        let second = build_outbox_batch(&conn, &first.cursor, 100, &none).unwrap();
        assert!(
            second
                .records
                .iter()
                .any(|r| r.kind == "reflection" && r.content.contains("amended after the run")),
            "revised trajectory must appear in the next batch: {:?}",
            second
                .records
                .iter()
                .map(|r| (r.kind.as_str(), r.content.as_str()))
                .collect::<Vec<_>>()
        );
        assert!(second.cursor.trajectory_updated_ms >= 50);
    }

    #[test]
    fn n_linked_commits_yield_n_session_outcome_envelopes() {
        let conn = mem();
        add_history_project(
            &conn,
            "s-n",
            Some("/Users/me/Projects/relayhistory"),
            "land the fix",
            1,
        );
        for i in 1..=3 {
            add_commit_link(&conn, "s-n", &format!("sha{i:03}"), "cwd+branch");
        }
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        let outcomes: Vec<_> = batch
            .records
            .iter()
            .filter(|r| r.kind == "session_outcome")
            .collect();
        assert_eq!(outcomes.len(), 3);
        let shas: Vec<_> = outcomes
            .iter()
            .map(|r| r.commit_sha.clone().unwrap())
            .collect();
        assert_eq!(shas, vec!["sha001", "sha002", "sha003"]);
        assert!(outcomes
            .iter()
            .all(|r| r.match_method.as_deref() == Some("cwd+branch")));
        assert!(outcomes
            .iter()
            .all(|r| r.project_id.as_deref() == Some("relayhistory")));
        assert!(outcomes.iter().all(|r| r.confidence == Some(0.9)));
        let v = serde_json::to_value(outcomes[0]).unwrap();
        assert_eq!(v["kind"], "session_outcome");
        assert_eq!(v["commitSha"], "sha001");
        assert_eq!(v["matchMethod"], "cwd+branch");
        assert_eq!(v["numstat"]["files"][0]["path"], "src/lib.rs");
        assert_eq!(v["files"][0], "src/lib.rs");
        assert_eq!(batch.cursor.commit_link_id, 3);

        let empty = build_outbox_batch(&conn, &batch.cursor, 100, &none).unwrap();
        assert!(!empty.records.iter().any(|r| r.kind == "session_outcome"));
    }

    #[test]
    fn session_outcome_shipped_at_comes_from_evidence_commit_time() {
        let conn = mem();
        add_history_project(
            &conn,
            "s-ship",
            Some("/Users/me/Projects/relayhistory"),
            "land it",
            1,
        );
        add_commit_link_evidence(
            &conn,
            "s-ship",
            "deadbeef",
            "cwd",
            Some(r#"{"commit_time_ms":1000}"#),
        );
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        let outcome = batch
            .records
            .iter()
            .find(|r| r.kind == "session_outcome")
            .unwrap();
        assert_eq!(
            outcome.shipped_at.as_deref(),
            Some("1970-01-01T00:00:01.000Z")
        );
        assert_eq!(outcome.ts, "2026-06-21T10:00:00.000Z");
    }

    #[test]
    fn session_outcome_event_ids_differ_when_match_method_differs() {
        let conn = mem();
        add_history_project(
            &conn,
            "s-dup",
            Some("/Users/me/Projects/relayhistory"),
            "same commit two ways",
            1,
        );
        add_commit_link(&conn, "s-dup", "abc123def", "cwd");
        add_commit_link(&conn, "s-dup", "abc123def", "cwd+branch");
        let none = HashSet::new();
        let batch = build_outbox_batch(&conn, &SyncCursor::default(), 100, &none).unwrap();
        let ids: Vec<_> = batch
            .records
            .iter()
            .filter(|r| r.kind == "session_outcome")
            .map(|r| r.event_id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec![
                "session_outcome:claude:s-dup:abc123def:cwd",
                "session_outcome:claude:s-dup:abc123def:cwd+branch",
            ]
        );
    }
}
