"""
Audit Logger
------------
Writes structured entries to a SQLite database for every attribution decision
and appeal. SQLite is built into Python — no extra dependencies needed.

Two entry types:
    "decision" — written by POST /submit
    "appeal"   — written by POST /appeal (added in a later milestone)

The database file is created automatically on first use at:
    provenance-guard/audit_log.db

Schema (audit_log table):
    id            INTEGER  primary key, auto-increment
    content_id    TEXT     the UUID generated at submission time
    creator_id    TEXT     from the request body
    timestamp     TEXT     ISO 8601 UTC
    entry_type    TEXT     "decision" | "appeal"
    attribution   TEXT     "ai" | "human" | "uncertain"
    confidence    REAL     fused score 0.0–1.0 (currently Signal 1 raw_score)
    llm_score     REAL     Signal 1 raw_score
    status        TEXT     "classified" | "under_review" | "resolved"
    raw_entry     TEXT     full JSON blob for forward compatibility
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

# ── Database path ─────────────────────────────────────────────────────────────
# Sits at the project root, one level above the app/ package.
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "audit_log.db")


# ── Schema ────────────────────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id  TEXT    NOT NULL,
    creator_id  TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    entry_type  TEXT    NOT NULL,
    attribution TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    llm_score   REAL    NOT NULL,
    status      TEXT    NOT NULL,
    raw_entry   TEXT    NOT NULL
);
"""


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the table exists."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


# ── Public API ────────────────────────────────────────────────────────────────


def log_decision(
    content_id: str,
    creator_id: str,
    attribution: str,
    confidence: float,
    llm_score: float,
    status: str = "classified",
) -> None:
    """
    Write a decision entry to the audit log.

    Called by POST /submit immediately before returning the response,
    so every submission is recorded regardless of what the caller does next.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # The raw_entry is the full structured record as JSON — useful for
    # forward compatibility when we add Signal 2 and more fields later.
    raw_entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "entry_type": "decision",
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "status": status,
    }

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO audit_log
                (content_id, creator_id, timestamp, entry_type,
                 attribution, confidence, llm_score, status, raw_entry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_id,
                creator_id,
                timestamp,
                "decision",
                attribution,
                confidence,
                llm_score,
                status,
                json.dumps(raw_entry),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_log(limit: int = 50) -> list[dict]:
    """
    Return the most recent *limit* audit log entries as a list of dicts.
    Used by GET /log (to be implemented) and for inspection during testing.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(row["raw_entry"]) for row in rows]
    finally:
        conn.close()
