"""Tests for ai-hist — 100% coverage target."""

import importlib.machinery
import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Import the ai-hist script (no .py extension)
_path = str(Path(__file__).parent / "ai-hist")
_loader = importlib.machinery.SourceFileLoader("ai_hist", _path)
_spec = importlib.util.spec_from_loader("ai_hist", _loader, origin=_path)
ai_hist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ai_hist)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Set up isolated DB, state, and source files."""
    db_path = tmp_path / "test.db"
    state_path = tmp_path / ".sync-state.json"
    claude_hist = tmp_path / "claude_history.jsonl"
    codex_hist = tmp_path / "codex_history.jsonl"

    monkeypatch.setattr(ai_hist, "DB_PATH", db_path)
    monkeypatch.setattr(ai_hist, "STATE_PATH", state_path)
    monkeypatch.setattr(ai_hist, "SOURCES", {
        "claude": claude_hist,
        "codex": codex_hist,
    })

    return SimpleNamespace(
        db_path=db_path,
        state_path=state_path,
        claude_hist=claude_hist,
        codex_hist=codex_hist,
        tmp_path=tmp_path,
    )


def make_claude_entry(display, timestamp=1700000000000, project="/proj", session_id="s1"):
    return json.dumps({
        "display": display,
        "timestamp": timestamp,
        "project": project,
        "sessionId": session_id,
        "pastedContents": {},
    })


def make_codex_entry(text, ts=1700000000, session_id="cs1"):
    return json.dumps({
        "text": text,
        "ts": ts,
        "session_id": session_id,
    })


def seed_db(env, claude_lines=None, codex_lines=None):
    """Write history files and run sync."""
    if claude_lines:
        env.claude_hist.write_text("\n".join(claude_lines) + "\n")
    if codex_lines:
        env.codex_hist.write_text("\n".join(codex_lines) + "\n")
    ai_hist.cmd_sync()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseClaude:
    def test_valid_entry(self):
        line = make_claude_entry("hello world", 1700000000000, "/my/project", "sess1")
        result = ai_hist.parse_claude(line)
        assert result == {
            "source": "claude",
            "session_id": "sess1",
            "project": "/my/project",
            "prompt": "hello world",
            "timestamp_ms": 1700000000000,
        }

    def test_empty_display_returns_none(self):
        line = json.dumps({"display": "", "timestamp": 123})
        assert ai_hist.parse_claude(line) is None

    def test_whitespace_display_returns_none(self):
        line = json.dumps({"display": "   ", "timestamp": 123})
        assert ai_hist.parse_claude(line) is None

    def test_missing_display_returns_none(self):
        line = json.dumps({"timestamp": 123})
        assert ai_hist.parse_claude(line) is None

    def test_missing_optional_fields(self):
        line = json.dumps({"display": "test"})
        result = ai_hist.parse_claude(line)
        assert result["session_id"] is None
        assert result["project"] is None
        assert result["timestamp_ms"] == 0


class TestParseCodex:
    def test_valid_entry(self):
        line = make_codex_entry("fix the bug", 1700000000, "cs1")
        result = ai_hist.parse_codex(line)
        assert result == {
            "source": "codex",
            "session_id": "cs1",
            "project": None,
            "prompt": "fix the bug",
            "timestamp_ms": 1700000000000,
        }

    def test_empty_text_returns_none(self):
        line = json.dumps({"text": "", "ts": 123})
        assert ai_hist.parse_codex(line) is None

    def test_whitespace_text_returns_none(self):
        line = json.dumps({"text": "  ", "ts": 100})
        assert ai_hist.parse_codex(line) is None

    def test_missing_text_returns_none(self):
        line = json.dumps({"ts": 123})
        assert ai_hist.parse_codex(line) is None

    def test_missing_optional_fields(self):
        line = json.dumps({"text": "hello"})
        result = ai_hist.parse_codex(line)
        assert result["session_id"] is None
        assert result["timestamp_ms"] == 0


# ---------------------------------------------------------------------------
# Core function tests
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_tables(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        ai_hist.init_db(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchone()
        assert tables is not None
        fts = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history_fts'"
        ).fetchone()
        assert fts is not None
        conn.close()

    def test_idempotent(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        ai_hist.init_db(conn)
        ai_hist.init_db(conn)  # should not raise
        conn.close()


class TestLoadSaveState:
    def test_load_empty(self, tmp_env):
        state = ai_hist.load_state()
        assert state == {}

    def test_save_and_load(self, tmp_env):
        ai_hist.save_state({"claude": 100, "codex": 200})
        state = ai_hist.load_state()
        assert state == {"claude": 100, "codex": 200}

    def test_save_creates_parent_dir(self, tmp_env, monkeypatch):
        new_state = tmp_env.tmp_path / "sub" / "dir" / ".sync-state.json"
        monkeypatch.setattr(ai_hist, "STATE_PATH", new_state)
        ai_hist.save_state({"x": 1})
        assert new_state.exists()


class TestFmtRow:
    def test_with_project(self):
        result = ai_hist.fmt_row(1, "claude", "/my/project", "hello", 1700000000000)
        assert "(claude)" in result
        assert "[/my/project]" in result
        assert "hello" in result
        assert "#1" in result

    def test_without_project(self):
        result = ai_hist.fmt_row(2, "codex", None, "world", 1700000000000)
        assert "(codex)" in result
        assert "[" not in result

    def test_long_prompt_truncated(self):
        long_prompt = "x" * 200
        result = ai_hist.fmt_row(3, "claude", None, long_prompt, 1700000000000)
        assert result.endswith("...")
        assert "x" * 120 in result

    def test_newlines_replaced(self):
        result = ai_hist.fmt_row(4, "claude", None, "line1\nline2", 1700000000000)
        assert "\n" not in result
        assert "line1 line2" in result

    def test_short_prompt_not_truncated(self):
        result = ai_hist.fmt_row(5, "claude", None, "short", 1700000000000)
        assert "..." not in result

    def test_verbose_no_truncation(self):
        long_prompt = "x" * 200
        result = ai_hist.fmt_row(6, "claude", None, long_prompt, 1700000000000, verbose=True)
        assert "..." not in result
        assert "x" * 200 in result

    def test_verbose_preserves_newlines(self):
        result = ai_hist.fmt_row(7, "claude", None, "line1\nline2", 1700000000000, verbose=True)
        assert "line1\nline2" in result


# ---------------------------------------------------------------------------
# Command tests
# ---------------------------------------------------------------------------

class TestCmdSync:
    def test_sync_claude_entries(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(
            make_claude_entry("first prompt", 1700000001000) + "\n"
            + make_claude_entry("second prompt", 1700000002000) + "\n"
        )
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "+2" in captured.out
        assert "Total: 2" in captured.out

    def test_sync_codex_entries(self, tmp_env, capsys):
        tmp_env.codex_hist.write_text(
            make_codex_entry("codex prompt", 1700000001) + "\n"
        )
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "+1" in captured.out

    def test_sync_both_sources(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(make_claude_entry("c1", 1700000001000) + "\n")
        tmp_env.codex_hist.write_text(make_codex_entry("x1", 1700000001) + "\n")
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "Total: 2" in captured.out

    def test_incremental_sync(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(make_claude_entry("first", 1700000001000) + "\n")
        ai_hist.cmd_sync()
        with open(tmp_env.claude_hist, "a") as f:
            f.write(make_claude_entry("second", 1700000002000) + "\n")
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "Total: 2" in captured.out

    def test_sync_up_to_date(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(make_claude_entry("first", 1700000001000) + "\n")
        ai_hist.cmd_sync()
        capsys.readouterr()
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "up to date" in captured.out

    def test_sync_missing_source(self, tmp_env, capsys):
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_sync_skips_empty_lines(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(
            make_claude_entry("one", 1700000001000) + "\n\n\n"
            + make_claude_entry("two", 1700000002000) + "\n"
        )
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "Total: 2" in captured.out

    def test_sync_handles_invalid_json(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(
            "not valid json\n"
            + make_claude_entry("valid", 1700000001000) + "\n"
        )
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "+1" in captured.out
        assert "1 errors" in captured.out

    def test_sync_skips_none_rows(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(
            json.dumps({"display": "", "timestamp": 123}) + "\n"
            + make_claude_entry("real", 1700000001000) + "\n"
        )
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "+1" in captured.out

    def test_sync_dedup_on_reinsert(self, tmp_env, capsys):
        tmp_env.claude_hist.write_text(make_claude_entry("dupe", 1700000001000) + "\n")
        ai_hist.cmd_sync()
        ai_hist.save_state({})
        ai_hist.cmd_sync()
        conn = sqlite3.connect(str(tmp_env.db_path))
        count = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        conn.close()
        assert count == 1

    def test_sync_creates_db_parent_dir(self, tmp_env, monkeypatch):
        nested = tmp_env.tmp_path / "a" / "b" / "test.db"
        monkeypatch.setattr(ai_hist, "DB_PATH", nested)
        ai_hist.cmd_sync()
        assert nested.exists()

    def test_sync_handles_sqlite_error_on_insert(self, tmp_env, capsys, monkeypatch):
        tmp_env.claude_hist.write_text(
            make_claude_entry("will fail", 1700000001000) + "\n"
            + make_claude_entry("also fails", 1700000002000) + "\n"
        )
        original_connect = sqlite3.connect

        class FaultyConnection:
            def __init__(self, conn):
                self._conn = conn
                self._initialized = False

            def executescript(self, sql):
                return self._conn.executescript(sql)

            def execute(self, sql, params=None):
                if sql.startswith("INSERT OR IGNORE INTO history") and self._initialized:
                    raise sqlite3.OperationalError("simulated error")
                result = self._conn.execute(sql, params) if params else self._conn.execute(sql)
                if "PRAGMA" in sql:
                    self._initialized = True
                return result

            def commit(self):
                return self._conn.commit()

            def close(self):
                return self._conn.close()

        def patched_connect(path):
            real_conn = original_connect(path)
            return FaultyConnection(real_conn)

        monkeypatch.setattr(sqlite3, "connect", patched_connect)
        ai_hist.cmd_sync()
        captured = capsys.readouterr()
        assert "2 errors" in captured.out


class TestCmdSearch:
    def test_search_finds_match(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("fix authentication bug", 1700000001000, "/proj"),
            make_claude_entry("add new feature", 1700000002000, "/proj"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(query=["authentication"], source=None, project=None, limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "authentication" in captured.out
        assert "#" in captured.out  # ID is shown

    def test_search_no_results(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("hello", 1700000001000)])
        capsys.readouterr()
        args = SimpleNamespace(query=["zzzznonexistent"], source=None, project=None, limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "No results." in captured.out

    def test_search_filter_by_source(self, tmp_env, capsys):
        seed_db(tmp_env,
            claude_lines=[make_claude_entry("shared term", 1700000001000)],
            codex_lines=[make_codex_entry("shared term", 1700000002)],
        )
        capsys.readouterr()
        args = SimpleNamespace(query=["shared"], source="codex", project=None, limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "(codex)" in captured.out
        assert "(claude)" not in captured.out

    def test_search_filter_by_project(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("in relay", 1700000001000, "/proj/relay"),
            make_claude_entry("in dashboard", 1700000002000, "/proj/dashboard"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(query=["in"], source=None, project="relay", limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "relay" in captured.out
        assert "dashboard" not in captured.out

    def test_search_respects_limit(self, tmp_env, capsys):
        lines = [make_claude_entry(f"test query {i}", 1700000000000 + i * 1000) for i in range(10)]
        seed_db(tmp_env, claude_lines=lines)
        capsys.readouterr()
        args = SimpleNamespace(query=["test"], source=None, project=None, limit=3)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        result_lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        assert len(result_lines) == 3

    def test_search_multi_word_query(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("fix the authentication bug", 1700000001000),
        ])
        capsys.readouterr()
        args = SimpleNamespace(query=["fix", "bug"], source=None, project=None, limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "authentication" in captured.out

    def test_search_hyphenated_term(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("deploy agent-relay to prod", 1700000001000, "/proj"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(query=["agent-relay"], source=None, project=None, limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "agent-relay" in captured.out


class TestCmdRecent:
    def test_recent_default(self, tmp_env, capsys):
        lines = [make_claude_entry(f"prompt {i}", 1700000000000 + i * 1000) for i in range(5)]
        seed_db(tmp_env, claude_lines=lines)
        capsys.readouterr()
        args = SimpleNamespace(n=20, source=None, project=None)
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        result_lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        assert len(result_lines) == 5

    def test_recent_limited(self, tmp_env, capsys):
        lines = [make_claude_entry(f"prompt {i}", 1700000000000 + i * 1000) for i in range(10)]
        seed_db(tmp_env, claude_lines=lines)
        capsys.readouterr()
        args = SimpleNamespace(n=3, source=None, project=None)
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        result_lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        assert len(result_lines) == 3

    def test_recent_order_newest_first(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("old prompt", 1700000001000),
            make_claude_entry("new prompt", 1700000099000),
        ])
        capsys.readouterr()
        args = SimpleNamespace(n=2, source=None, project=None)
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "new prompt" in lines[0]
        assert "old prompt" in lines[1]

    def test_recent_empty_db(self, tmp_env, capsys):
        seed_db(tmp_env)
        capsys.readouterr()
        args = SimpleNamespace(n=10, source=None, project=None)
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_recent_filter_by_source(self, tmp_env, capsys):
        seed_db(tmp_env,
            claude_lines=[make_claude_entry("claude msg", 1700000001000)],
            codex_lines=[make_codex_entry("codex msg", 1700000002)],
        )
        capsys.readouterr()
        args = SimpleNamespace(n=20, source="claude", project=None)
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        assert "(claude)" in captured.out
        assert "(codex)" not in captured.out

    def test_recent_filter_by_project(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("in relay", 1700000001000, "/proj/relay"),
            make_claude_entry("in dash", 1700000002000, "/proj/dashboard"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(n=20, source=None, project="dashboard")
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        assert "dash" in captured.out
        assert "relay" not in captured.out

    def test_recent_filter_by_source_and_project(self, tmp_env, capsys):
        seed_db(tmp_env,
            claude_lines=[make_claude_entry("c relay", 1700000001000, "/proj/relay")],
            codex_lines=[make_codex_entry("x msg", 1700000002)],
        )
        capsys.readouterr()
        args = SimpleNamespace(n=20, source="claude", project="relay")
        ai_hist.cmd_recent(args)
        captured = capsys.readouterr()
        assert "c relay" in captured.out
        assert len([l for l in captured.out.strip().split("\n") if l.strip()]) == 1


class TestCmdShow:
    def test_show_existing_entry(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("full prompt text here\nwith newlines", 1700000001000, "/proj/x", "sess-abc"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(id=1)
        ai_hist.cmd_show(args)
        captured = capsys.readouterr()
        assert "ID:" in captured.out
        assert "Source:    claude" in captured.out
        assert "Session:   sess-abc" in captured.out
        assert "Project:   /proj/x" in captured.out
        assert "full prompt text here\nwith newlines" in captured.out

    def test_show_nonexistent_entry(self, tmp_env, capsys):
        seed_db(tmp_env)
        capsys.readouterr()
        args = SimpleNamespace(id=999)
        ai_hist.cmd_show(args)
        captured = capsys.readouterr()
        assert "No entry with id 999" in captured.out

    def test_show_entry_without_session_or_project(self, tmp_env, capsys):
        # Use a codex entry with no session_id to test (none) display
        tmp_env.codex_hist.write_text(
            json.dumps({"text": "nosession", "ts": 1700000001}) + "\n"
        )
        ai_hist.cmd_sync()
        capsys.readouterr()
        args = SimpleNamespace(id=1)
        ai_hist.cmd_show(args)
        captured = capsys.readouterr()
        assert "Session:   (none)" in captured.out
        assert "Project:   (none)" in captured.out


class TestCmdSession:
    def test_session_shows_all_prompts(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("first in session", 1700000001000, "/proj", "sess-xyz"),
            make_claude_entry("second in session", 1700000002000, "/proj", "sess-xyz"),
            make_claude_entry("different session", 1700000003000, "/proj", "sess-other"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(session_id="sess-xyz", full=False)
        ai_hist.cmd_session(args)
        captured = capsys.readouterr()
        assert "sess-xyz" in captured.out
        assert "2 entries" in captured.out
        assert "first in session" in captured.out
        assert "second in session" in captured.out
        assert "different session" not in captured.out

    def test_session_not_found(self, tmp_env, capsys):
        seed_db(tmp_env)
        capsys.readouterr()
        args = SimpleNamespace(session_id="nonexistent", full=False)
        ai_hist.cmd_session(args)
        captured = capsys.readouterr()
        assert "No entries for session nonexistent" in captured.out

    def test_session_full_flag(self, tmp_env, capsys):
        long_prompt = "x" * 200
        seed_db(tmp_env, claude_lines=[
            make_claude_entry(long_prompt, 1700000001000, "/proj", "sess-full"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(session_id="sess-full", full=True)
        ai_hist.cmd_session(args)
        captured = capsys.readouterr()
        assert "x" * 200 in captured.out
        assert "..." not in captured.out

    def test_session_chronological_order(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("later", 1700000099000, "/proj", "sess-order"),
            make_claude_entry("earlier", 1700000001000, "/proj", "sess-order"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(session_id="sess-order", full=False)
        ai_hist.cmd_session(args)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if l.strip() and "(" in l and "#" in l]
        assert "earlier" in lines[0]
        assert "later" in lines[1]


class TestCmdStats:
    def test_stats_with_data(self, tmp_env, capsys):
        seed_db(tmp_env,
            claude_lines=[
                make_claude_entry("c1", 1700000001000, "/proj/a"),
                make_claude_entry("c2", 1700000002000, "/proj/b"),
            ],
            codex_lines=[
                make_codex_entry("x1", 1700000003),
            ],
        )
        capsys.readouterr()
        ai_hist.cmd_stats()
        captured = capsys.readouterr()
        assert "Total entries: 3" in captured.out
        assert "claude: 2" in captured.out
        assert "codex: 1" in captured.out
        assert "Date range:" in captured.out
        assert "/proj/a" in captured.out or "/proj/b" in captured.out

    def test_stats_empty_db(self, tmp_env, capsys):
        seed_db(tmp_env)
        capsys.readouterr()
        ai_hist.cmd_stats()
        captured = capsys.readouterr()
        assert "Total entries: 0" in captured.out
        assert "Date range:" not in captured.out

    def test_stats_no_projects(self, tmp_env, capsys):
        seed_db(tmp_env, codex_lines=[make_codex_entry("x1", 1700000001)])
        capsys.readouterr()
        ai_hist.cmd_stats()
        captured = capsys.readouterr()
        assert "Top 10 projects:" in captured.out


class TestCmdWatch:
    def test_watch_runs_sync_and_stops(self, tmp_env, capsys):
        call_count = 0

        def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt()

        args = SimpleNamespace(interval=1)
        with patch.object(time, "sleep", mock_sleep):
            with pytest.raises(KeyboardInterrupt):
                ai_hist.cmd_watch(args)
        captured = capsys.readouterr()
        assert "Watching every 1s" in captured.out

    def test_watch_handles_sync_error(self, tmp_env, capsys):
        call_count = 0

        def failing_sync(args=None):
            raise RuntimeError("test error")

        def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt()

        args = SimpleNamespace(interval=5)
        with patch.object(ai_hist, "cmd_sync", failing_sync):
            with patch.object(time, "sleep", mock_sleep):
                with pytest.raises(KeyboardInterrupt):
                    ai_hist.cmd_watch(args)
        captured = capsys.readouterr()
        assert "Error: test error" in captured.err


# ---------------------------------------------------------------------------
# CLI / main tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_args_prints_help(self, capsys):
        with patch("sys.argv", ["ai-hist"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "Sync & search" in captured.out

    def test_sync_command(self, tmp_env, capsys):
        with patch("sys.argv", ["ai-hist", "sync"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "Total:" in captured.out

    def test_search_command(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("hello world", 1700000001000)])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "search", "hello"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_recent_command(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("recent test", 1700000001000)])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "recent", "5"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "recent test" in captured.out

    def test_stats_command(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("stats test", 1700000001000)])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "stats"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "Total entries: 1" in captured.out

    def test_search_with_source_flag(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("flagtest", 1700000001000)])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "search", "flagtest", "--source", "claude", "--limit", "5"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "flagtest" in captured.out

    def test_search_with_project_flag(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("in relay", 1700000001000, "/proj/relay"),
            make_claude_entry("in dash", 1700000002000, "/proj/dash"),
        ])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "search", "in", "--project", "relay"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "relay" in captured.out
        assert "dash" not in captured.out

    def test_show_command(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("show me", 1700000001000)])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "show", "1"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "show me" in captured.out

    def test_session_command(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("sess prompt", 1700000001000, "/p", "s1")])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "session", "s1"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "sess prompt" in captured.out

    def test_session_command_with_full(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("x" * 200, 1700000001000, "/p", "s2")])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "session", "s2", "--full"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "x" * 200 in captured.out

    def test_watch_command_dispatches(self, tmp_env):
        def mock_watch(args):
            assert args.interval == 60

        with patch.object(ai_hist, "cmd_watch", mock_watch):
            with patch("sys.argv", ["ai-hist", "watch"]):
                ai_hist.main()

    def test_recent_with_source_and_project(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[make_claude_entry("filtered", 1700000001000, "/proj/x")])
        capsys.readouterr()
        with patch("sys.argv", ["ai-hist", "recent", "5", "--source", "claude", "--project", "proj"]):
            ai_hist.main()
        captured = capsys.readouterr()
        assert "filtered" in captured.out


# ---------------------------------------------------------------------------
# FTS trigger integration test
# ---------------------------------------------------------------------------

class TestFTSIntegration:
    def test_fts_index_populated_on_insert(self, tmp_env):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("unique searchable term xyzzy", 1700000001000, "/proj"),
        ])
        conn = sqlite3.connect(str(tmp_env.db_path))
        rows = conn.execute(
            "SELECT rowid FROM history_fts WHERE history_fts MATCH 'xyzzy'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1

    def test_fts_searches_project_field(self, tmp_env, capsys):
        seed_db(tmp_env, claude_lines=[
            make_claude_entry("some prompt", 1700000001000, "/unique/project/path"),
        ])
        capsys.readouterr()
        args = SimpleNamespace(query=["unique"], source=None, project=None, limit=20)
        ai_hist.cmd_search(args)
        captured = capsys.readouterr()
        assert "some prompt" in captured.out
