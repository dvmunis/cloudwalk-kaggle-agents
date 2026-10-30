"""Attempt trace: every agent decision and tool call of an orchestrated attempt.

An *attempt* is one end-to-end run of the agent team on a competition. Events
are appended by the agents and by the worker itself (tool calls and their
results). A decision event carries what the agent decided and why, the full
context it saw, and the LLM call's model, attempts, latency and tokens, so a
decision can be audited from the trace alone. Stored in SQLite under RUNS_DIR
and exported to JSONL when the attempt finishes.

Attempts whose orchestrator died without reporting (n8n restarted, the error
workflow failed) would stay 'running' forever; any attempt with no event for
STALE_ATTEMPT_MINUTES is closed as 'orphaned' when attempts are read.
"""

import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone

from . import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id   TEXT PRIMARY KEY,
    competition  TEXT NOT NULL,
    status       TEXT NOT NULL,
    goal         TEXT,
    execution_id TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    summary     TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    ts         TEXT NOT NULL,
    agent      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    reasoning  TEXT,
    context    TEXT,
    meta       TEXT
);
"""

EVENT_KINDS = ("decision", "tool_call", "tool_result", "message", "error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    path = settings.runs_dir() / "trace.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Databases created before these columns existed.
    for table, column in (("attempts", "execution_id"), ("events", "context"), ("events", "meta")):
        if column not in {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
    return conn


def _stale_after() -> timedelta:
    # Longer than the slowest step (a full-data training run takes up to ~10 minutes).
    return timedelta(minutes=float(os.getenv("STALE_ATTEMPT_MINUTES", "30")))


def start_attempt(competition: str, goal: str = "", execution_id: str | None = None) -> dict:
    attempt_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO attempts (attempt_id, competition, status, goal, execution_id, started_at) "
            "VALUES (?, ?, 'running', ?, ?, ?)",
            (attempt_id, competition, goal, execution_id, _now()),
        )
    return get_attempt(attempt_id)


def fail_execution(execution_id: str, node: str | None, error: str) -> dict:
    """Mark the attempt started by an orchestrator execution as failed, recording where it broke."""
    with closing(_connect()) as conn:
        row = conn.execute("SELECT attempt_id FROM attempts WHERE execution_id = ?", (execution_id,)).fetchone()
    if row is None:
        raise ValueError(f"No attempt for execution '{execution_id}'")
    log_event(row["attempt_id"], "orchestrator", "error", {"node": node, "error": error})
    return finish_attempt(row["attempt_id"], "failed", {"failed_node": node, "error": error})


def log_event(attempt_id: str, agent: str, kind: str, content, reasoning: str | None = None,
              context=None, meta: dict | None = None) -> None:
    if kind not in EVENT_KINDS:
        raise ValueError(f"Unknown event kind '{kind}'. Use one of {EVENT_KINDS}")
    with closing(_connect()) as conn, conn:
        if not conn.execute("SELECT 1 FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone():
            raise ValueError(f"Unknown attempt_id '{attempt_id}'")
        conn.execute(
            "INSERT INTO events (attempt_id, ts, agent, kind, content, reasoning, context, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (attempt_id, _now(), agent, kind, json.dumps(content, default=str), reasoning,
             None if context is None else json.dumps(context, default=str),
             None if meta is None else json.dumps(meta, default=str)),
        )


def finish_attempt(attempt_id: str, status: str, summary: dict) -> dict:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE attempts SET status = ?, finished_at = ?, summary = ? WHERE attempt_id = ?",
            (status, _now(), json.dumps(summary, default=str), attempt_id),
        )
    attempt = get_attempt(attempt_id)
    _export_jsonl(attempt)
    return attempt


def _row_to_attempt(row: sqlite3.Row) -> dict:
    attempt = dict(row)
    attempt["summary"] = json.loads(attempt["summary"]) if attempt["summary"] else None
    return attempt


def sweep_stale(now: datetime | None = None) -> list[str]:
    """Close running attempts with no event for longer than the stale limit as 'orphaned'."""
    now = now or datetime.now(timezone.utc)
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT a.attempt_id, a.started_at, MAX(e.ts) AS last_event FROM attempts a "
            "LEFT JOIN events e ON e.attempt_id = a.attempt_id WHERE a.status = 'running' GROUP BY a.attempt_id"
        ).fetchall()
    orphaned = []
    for row in rows:
        last = row["last_event"] or row["started_at"]
        if now - datetime.fromisoformat(last) > _stale_after():
            reason = "no event since the orchestrator's last report; it stopped without closing the attempt"
            log_event(row["attempt_id"], "worker", "error", {"error": reason, "last_event_at": last})
            finish_attempt(row["attempt_id"], "orphaned", {"error": reason, "last_event_at": last})
            orphaned.append(row["attempt_id"])
    return orphaned


def get_attempt(attempt_id: str) -> dict:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown attempt_id '{attempt_id}'")
        events = conn.execute("SELECT * FROM events WHERE attempt_id = ? ORDER BY id", (attempt_id,)).fetchall()
    attempt = _row_to_attempt(row)
    attempt["events"] = [
        {**dict(e), **{k: json.loads(e[k]) if e[k] is not None else None for k in ("content", "context", "meta")}}
        for e in events
    ]
    return attempt


def list_attempts(competition: str | None = None) -> list[dict]:
    query, params = "SELECT * FROM attempts", ()
    if competition:
        query, params = query + " WHERE competition = ?", (competition,)
    with closing(_connect()) as conn:
        rows = conn.execute(query + " ORDER BY started_at DESC", params).fetchall()
    return [_row_to_attempt(r) for r in rows]


def _export_jsonl(attempt: dict) -> None:
    """Mirror a closed attempt into the run folder, so the evidence travels without trace.db:
    its events as <attempt_id>.jsonl and one line per attempt (status, summary) in index.jsonl."""
    folder = settings.competition_runs_dir(attempt["competition"]) / "attempts"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{attempt['attempt_id']}.jsonl").open("w") as f:
        for event in attempt["events"]:
            f.write(json.dumps(event, default=str) + "\n")
    with (folder / "index.jsonl").open("w") as f:
        for row in reversed(list_attempts(attempt["competition"])):
            f.write(json.dumps(row, default=str) + "\n")
