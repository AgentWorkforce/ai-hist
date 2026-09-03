//! Conversation turns: the readable transcript half of cloud sync.
//!
//! The convergence outbox ([`crate::outbox`]) publishes *prompts* — what the human typed
//! — plus distilled trajectory records. It has never carried the other side of the
//! conversation. Measured on one machine on 2026-09-03, the cloud held 337,139 prompts
//! while these were local-only:
//!
//! ```text
//! assistant text      190,165   the answers
//! assistant tool_use  350,551   what the agent actually did
//! tool_result         338,614
//! assistant thinking    4,823   the reasoning
//! ```
//!
//! So "replay this session" returned only what the user typed. This module builds the
//! batches that `POST /v1/sessions/:sessionId/turns` needs to close that, from the local
//! `session_events` table the `ai-hist events` replay already reads.
//!
//! Network I/O lives in the binding layer, per the no-async-in-core rule; this module is
//! sync rusqlite reads and is unit-testable without a server.

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::collections::HashSet;

/// The server caps a turns request at 1,000. Chunk below it rather than at it, so a
/// future server-side reduction does not silently start rejecting whole sessions.
pub const MAX_TURNS_PER_REQUEST: usize = 500;

/// Sessions published per push run. A push already carries the convergence batch; this
/// bounds the extra work so a machine with a large backlog drains steadily instead of
/// stalling one run for hours.
pub const DEFAULT_SESSION_BUDGET: usize = 12;

/// One turn in the wire shape `POST /v1/sessions/:sessionId/turns` accepts.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ConversationTurn {
    #[serde(rename = "sessionOwner")]
    pub session_owner: String,
    #[serde(rename = "turnIndex")]
    pub turn_index: i64,
    /// `user` | `assistant` | `system` — the only roles the server accepts.
    pub role: String,
    pub content: String,
    #[serde(rename = "actorName")]
    pub actor_name: String,
    /// `owner` | `steerer`.
    #[serde(rename = "actorRole")]
    pub actor_role: String,
    pub metadata: serde_json::Value,
    /// RFC3339.
    pub ts: String,
}

/// A whole session's turns, chunked for the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionTurns {
    pub session_id: String,
    pub source: String,
    /// Chunks of at most [`MAX_TURNS_PER_REQUEST`], in turn order.
    pub chunks: Vec<Vec<ConversationTurn>>,
}

/// The batch a single push run should publish, plus the watermark it advances to.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnsBatch {
    pub sessions: Vec<SessionTurns>,
    /// Highest `session_events.id` scanned, emitted or not.
    pub session_event_id: i64,
}

struct RawEvent {
    session_id: String,
    source: String,
    ts_ms: i64,
    role: String,
    kind: String,
    text: Option<String>,
    model: Option<String>,
}

/// Map a local `session_events` role/kind pair onto the three roles the server accepts.
///
/// `tool_result` has no server-side role of its own and is not the assistant speaking, so
/// it lands as `system` with the original kind preserved in `metadata`. Dropping these
/// would leave a transcript where the agent calls a tool and nothing ever comes back.
fn wire_role(role: &str) -> &'static str {
    match role {
        "user" => "user",
        "assistant" => "assistant",
        _ => "system",
    }
}

/// Who to attribute a turn to. Never empty — the server rejects an empty `actorName`, and
/// an unattributed turn in a shared transcript is worse than a coarse one.
fn actor_name(role: &str, model: Option<&str>, source: &str) -> String {
    match role {
        "user" => "user".to_string(),
        "assistant" => model
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(source)
            .to_string(),
        _ => format!("{source}:tool"),
    }
}

fn epoch_ms_to_iso(ms: i64) -> String {
    crate::convergence::epoch_ms_to_iso(ms)
}

/// Build the next batch of conversation turns past `session_event_id`.
///
/// Sessions are published **whole**, not incrementally, because `turnIndex` is the
/// server's idempotency key and it is a position within the session. Publishing only the
/// new tail would number those turns from zero and overwrite the beginning of the
/// transcript with its own end — a corruption that still returns 200 and still looks like
/// a full session on read. Whole-session publishing costs a re-send and cannot do that.
///
/// `incognito` sessions are skipped but still advance the watermark, matching the
/// convergence outbox so a suppressed session is never rescanned forever.
pub fn build_turns_batch(
    conn: &Connection,
    session_event_id: i64,
    session_budget: usize,
    incognito: &HashSet<String>,
) -> Result<TurnsBatch> {
    let budget = session_budget.max(1);

    // Which sessions have anything new, oldest change first so the backlog drains in a
    // predictable order rather than jumping around.
    let mut stmt = conn.prepare(
        "SELECT session_id, source, MIN(id) AS first_new_id \
         FROM session_events WHERE id > ?1 \
         GROUP BY session_id, source ORDER BY first_new_id ASC LIMIT ?2",
    )?;
    let pending: Vec<(String, String)> = stmt
        .query_map(rusqlite::params![session_event_id, budget as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?
        .filter_map(|row| row.ok())
        .collect();

    if pending.is_empty() {
        return Ok(TurnsBatch {
            sessions: Vec::new(),
            session_event_id,
        });
    }

    // Advance to the highest id belonging to the sessions actually taken this run. Using
    // the table-wide MAX would skip every session whose events sort after the budget cut.
    let mut watermark = session_event_id;
    let mut sessions = Vec::new();

    let mut events_stmt = conn.prepare(
        "SELECT session_id, source, ts_ms, role, kind, text, model, id \
         FROM session_events WHERE session_id = ?1 AND source = ?2 \
         ORDER BY ts_ms ASC, id ASC",
    )?;

    for (session_id, source) in pending {
        let rows = events_stmt.query_map(rusqlite::params![&session_id, &source], |r| {
            Ok((
                RawEvent {
                    session_id: r.get(0)?,
                    source: r.get(1)?,
                    ts_ms: r.get(2)?,
                    role: r.get(3)?,
                    kind: r.get(4)?,
                    text: r.get(5)?,
                    model: r.get(6)?,
                },
                r.get::<_, i64>(7)?,
            ))
        })?;

        let mut turns: Vec<ConversationTurn> = Vec::new();
        for row in rows {
            let (event, id) = match row {
                Ok(value) => value,
                Err(_) => continue,
            };
            watermark = watermark.max(id);

            // An event with no text carries nothing a reader can use. Skip it, but only
            // after the watermark has moved past it.
            let content = match event.text.as_deref().map(str::trim) {
                Some(text) if !text.is_empty() => text.to_string(),
                _ => continue,
            };

            let index = turns.len() as i64;
            turns.push(ConversationTurn {
                session_owner: event.source.clone(),
                turn_index: index,
                role: wire_role(&event.role).to_string(),
                content,
                actor_name: actor_name(&event.role, event.model.as_deref(), &event.source),
                actor_role: "owner".to_string(),
                metadata: serde_json::json!({
                    // The server validates `nativeCli` against claude/codex and ignores
                    // other keys, so the local kind survives for readers that want to
                    // distinguish thinking from a tool call.
                    "kind": event.kind,
                    "sourceRole": event.role,
                }),
                ts: epoch_ms_to_iso(event.ts_ms),
            });
            let _ = event.session_id;
        }

        if incognito.contains(&session_id) || turns.is_empty() {
            continue;
        }

        let chunks: Vec<Vec<ConversationTurn>> = turns
            .chunks(MAX_TURNS_PER_REQUEST)
            .map(|chunk| chunk.to_vec())
            .collect();

        sessions.push(SessionTurns {
            session_id,
            source,
            chunks,
        });
    }

    Ok(TurnsBatch {
        sessions,
        session_event_id: watermark,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                project TEXT, cwd TEXT, git_branch TEXT,
                message_id TEXT, parent_id TEXT,
                ts_ms INTEGER NOT NULL,
                role TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT, model TEXT, token_json TEXT,
                event_uid TEXT NOT NULL
            );",
        )
        .unwrap();
        conn
    }

    fn insert(conn: &Connection, session: &str, ts: i64, role: &str, kind: &str, text: &str) {
        conn.execute(
            "INSERT INTO session_events (source, session_id, ts_ms, role, kind, text, model, event_uid)
             VALUES ('claude', ?1, ?2, ?3, ?4, ?5, 'claude-opus-5', ?6)",
            rusqlite::params![session, ts, role, kind, text, format!("{session}-{ts}-{kind}")],
        )
        .unwrap();
    }

    #[test]
    fn publishes_both_sides_of_the_conversation() {
        let conn = db();
        insert(&conn, "s1", 1000, "user", "text", "why is it failing?");
        insert(
            &conn,
            "s1",
            2000,
            "assistant",
            "thinking",
            "consider the regex",
        );
        insert(&conn, "s1", 3000, "assistant", "tool_use", "grep -n scrub");
        insert(
            &conn,
            "s1",
            4000,
            "tool_result",
            "tool_result",
            "scrub.ts:41",
        );
        insert(
            &conn,
            "s1",
            5000,
            "assistant",
            "text",
            "the scheme is unbounded",
        );

        let batch = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();
        let turns = &batch.sessions[0].chunks[0];

        assert_eq!(turns.len(), 5);
        assert_eq!(
            turns.iter().map(|t| t.role.as_str()).collect::<Vec<_>>(),
            ["user", "assistant", "assistant", "system", "assistant"],
        );
        // Dense, ordered indices — the server's idempotency key.
        assert_eq!(
            turns.iter().map(|t| t.turn_index).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4],
        );
    }

    #[test]
    fn orders_by_timestamp_not_insertion() {
        let conn = db();
        insert(&conn, "s1", 3000, "assistant", "text", "third");
        insert(&conn, "s1", 1000, "user", "text", "first");
        insert(&conn, "s1", 2000, "assistant", "thinking", "second");

        let batch = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();
        let contents: Vec<&str> = batch.sessions[0].chunks[0]
            .iter()
            .map(|t| t.content.as_str())
            .collect();

        assert_eq!(contents, ["first", "second", "third"]);
    }

    /// `turnIndex` is a position in the session and the server's conflict key. Publishing
    /// only a new tail would renumber it from zero and overwrite the start of the
    /// transcript with its end — while still returning 200 and still reading as a full
    /// session. Whole-session publishing is what prevents that.
    #[test]
    fn republishes_the_whole_session_when_only_the_tail_is_new() {
        let conn = db();
        insert(&conn, "s1", 1000, "user", "text", "first");
        insert(&conn, "s1", 2000, "assistant", "text", "second");
        let first = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();
        assert_eq!(first.sessions[0].chunks[0].len(), 2);

        insert(&conn, "s1", 3000, "user", "text", "third");
        let second = build_turns_batch(&conn, first.session_event_id, 10, &HashSet::new()).unwrap();
        let turns = &second.sessions[0].chunks[0];

        assert_eq!(turns.len(), 3, "the whole session must be republished");
        assert_eq!(turns[0].content, "first");
        assert_eq!(turns[0].turn_index, 0);
        assert_eq!(turns[2].content, "third");
        assert_eq!(turns[2].turn_index, 2);
    }

    #[test]
    fn chunks_a_long_session_below_the_server_cap() {
        let conn = db();
        for i in 0..(MAX_TURNS_PER_REQUEST + 25) {
            insert(&conn, "s1", 1000 + i as i64, "user", "text", "hello");
        }

        let batch = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();
        let chunks = &batch.sessions[0].chunks;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), MAX_TURNS_PER_REQUEST);
        assert_eq!(chunks[1].len(), 25);
        // Indices stay absolute across the chunk boundary, or the second request
        // overwrites the first at indices 0..25.
        assert_eq!(chunks[1][0].turn_index, MAX_TURNS_PER_REQUEST as i64);
    }

    #[test]
    fn skips_empty_text_but_still_advances_the_watermark() {
        let conn = db();
        insert(&conn, "s1", 1000, "assistant", "tool_use", "   ");
        insert(&conn, "s1", 2000, "user", "text", "real");

        let batch = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();

        assert_eq!(batch.sessions[0].chunks[0].len(), 1);
        assert_eq!(
            batch.session_event_id, 2,
            "watermark must pass the skipped row"
        );
    }

    #[test]
    fn incognito_sessions_are_skipped_but_do_not_stall_the_watermark() {
        let conn = db();
        insert(&conn, "secret", 1000, "user", "text", "private");
        insert(&conn, "s2", 2000, "user", "text", "public");

        let incognito: HashSet<String> = ["secret".to_string()].into_iter().collect();
        let batch = build_turns_batch(&conn, 0, 10, &incognito).unwrap();

        let ids: Vec<&str> = batch
            .sessions
            .iter()
            .map(|s| s.session_id.as_str())
            .collect();
        assert_eq!(ids, ["s2"]);
        assert_eq!(batch.session_event_id, 2);
    }

    /// The watermark must cover the sessions actually taken. A table-wide MAX would skip
    /// every session sorting after the budget cut — they would never be published, and
    /// the run would still report success.
    #[test]
    fn budget_does_not_skip_sessions_past_the_cut() {
        let conn = db();
        insert(&conn, "s1", 1000, "user", "text", "one");
        insert(&conn, "s2", 2000, "user", "text", "two");
        insert(&conn, "s3", 3000, "user", "text", "three");

        let first = build_turns_batch(&conn, 0, 1, &HashSet::new()).unwrap();
        assert_eq!(first.sessions.len(), 1);
        assert_eq!(first.sessions[0].session_id, "s1");

        let second = build_turns_batch(&conn, first.session_event_id, 1, &HashSet::new()).unwrap();
        assert_eq!(
            second.sessions[0].session_id, "s2",
            "s2 must not be skipped"
        );

        let third = build_turns_batch(&conn, second.session_event_id, 1, &HashSet::new()).unwrap();
        assert_eq!(third.sessions[0].session_id, "s3");
    }

    #[test]
    fn attributes_every_turn_to_a_non_empty_actor() {
        let conn = db();
        insert(&conn, "s1", 1000, "user", "text", "hi");
        insert(&conn, "s1", 2000, "assistant", "text", "hello");
        insert(&conn, "s1", 3000, "tool_result", "tool_result", "ok");

        let batch = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();

        for turn in &batch.sessions[0].chunks[0] {
            assert!(
                !turn.actor_name.trim().is_empty(),
                "the server rejects an empty actorName: {turn:?}",
            );
        }
    }

    #[test]
    fn no_new_events_is_a_no_op_that_holds_its_watermark() {
        let conn = db();
        insert(&conn, "s1", 1000, "user", "text", "hi");
        let first = build_turns_batch(&conn, 0, 10, &HashSet::new()).unwrap();

        let second = build_turns_batch(&conn, first.session_event_id, 10, &HashSet::new()).unwrap();

        assert!(second.sessions.is_empty());
        assert_eq!(second.session_event_id, first.session_event_id);
    }
}
