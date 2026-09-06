use crate::cloud;
use anyhow::{Context, Result};
use serde_json::Value;
use std::io::{self, Write};
use std::path::Path;

pub fn run(
    session_id: &str,
    base_url: Option<&str>,
    limit: Option<usize>,
    max_content: Option<usize>,
    json: bool,
    out: Option<&Path>,
) -> Result<()> {
    let base_url = base_url
        .map(String::from)
        .unwrap_or_else(cloud::default_base_url);
    let auth = cloud::load_auth(Some(&base_url))?.context(
        "not authenticated for the selected stage — run `ai-hist login` or `ai-hist admin-mint` first",
    )?;
    let events = cloud::replay_events(&auth, session_id, limit, max_content)?;
    let body = if json {
        format!("{}\n", serde_json::to_string(&events)?)
    } else {
        render(session_id, &events)
    };
    // Finish fetching before touching the destination: an expired token on page two
    // must not overwrite an existing offline transcript with only page one.
    if let Some(path) = out {
        std::fs::write(path, body)
            .with_context(|| format!("writing replay to {}", path.display()))?;
    } else {
        io::stdout().lock().write_all(body.as_bytes())?;
    }
    Ok(())
}

fn render(session_id: &str, events: &[Value]) -> String {
    let mut output = format!(
        "Session {session_id} — {} event(s), oldest first\n",
        events.len()
    );
    if events.is_empty() {
        output.push_str("No events found.\n");
    }
    for event in events {
        let text = |field: &str| event[field].as_str().unwrap_or("");
        output.push_str(&format!(
            "\n[{}] {} / {} ({})\n",
            text("ts"),
            text("source"),
            text("kind"),
            text("eventId")
        ));
        for (field, label) in [
            ("actorName", "Actor"),
            ("actorRole", "Role"),
            ("toolName", "Tool"),
            ("taskTitle", "Task"),
        ] {
            if let Some(value) = event[field].as_str().filter(|value| !value.is_empty()) {
                output.push_str(&format!("{label}: {value}\n"));
            }
        }
        // The flag is authoritative even if content is empty or lacks the server's
        // inline suffix; a cut transcript must never look like the session ended here.
        if event["contentTruncated"].as_bool() == Some(true) {
            output.push_str("[CONTENT TRUNCATED by server maxContent; this event is incomplete]\n");
        }
        output.push_str(event["content"].as_str().unwrap_or("(no content)"));
        output.push('\n');
    }
    output
}
