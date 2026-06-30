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
    confidence    REAL     fused score 0.0-1.0
    llm_score     REAL     Signal A raw score (0.0-1.0)
    status        TEXT     "classified" | "under_review" | "resolved"
    raw_entry     TEXT     full JSON blob — includes stylo_score, sub_scores,
                           and any future fields without needing a schema change
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
    stylo_score: float,
    llm_verdict: str,
    stylo_verdict: str,
    status: str = "classified",
) -> None:
    """
    Write a decision entry to the audit log.

    Called by POST /submit immediately before returning the response,
    so every submission is recorded regardless of what the caller does next.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    raw_entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "entry_type": "decision",
        "attribution": attribution,
        "confidence": confidence,
        "signals": {
            "llm": {"score": llm_score, "verdict": llm_verdict},
            "stylo": {"score": stylo_score, "verdict": stylo_verdict},
        },
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
                llm_score,  # dedicated column for quick querying
                status,
                json.dumps(raw_entry),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_analytics() -> dict:
    """
    Compute aggregate metrics over the audit log for the analytics dashboard.

    Returns:
        {
            "total_submissions":     int,
            "detection_pattern": {
                "ai":        int,
                "human":     int,
                "uncertain": int,
                "ai_pct":        float,
                "human_pct":     float,
                "uncertain_pct": float
            },
            "appeal_rate": {
                "total_appeals":   int,
                "total_decisions": int,
                "rate_pct":        float
            },
            "avg_confidence_by_attribution": {
                "ai":        float | None,
                "human":     float | None,
                "uncertain": float | None
            },
            "signal_agreement_rate": {
                "agree_count":    int,
                "disagree_count": int,
                "rate_pct":       float
            }
        }
    """
    conn = _get_connection()
    try:
        decisions = conn.execute(
            "SELECT raw_entry FROM audit_log WHERE entry_type = 'decision'"
        ).fetchall()
        appeals = conn.execute(
            "SELECT raw_entry FROM audit_log WHERE entry_type = 'appeal'"
        ).fetchall()
    finally:
        conn.close()

    decision_entries = [json.loads(row["raw_entry"]) for row in decisions]
    appeal_entries = [json.loads(row["raw_entry"]) for row in appeals]

    total = len(decision_entries)

    # ── Detection pattern ─────────────────────────────────────────────────────
    counts = {"ai": 0, "human": 0, "uncertain": 0}
    for entry in decision_entries:
        attribution = entry.get("attribution", "uncertain")
        counts[attribution] = counts.get(attribution, 0) + 1

    def pct(n: int) -> float:
        return round((n / total) * 100, 2) if total else 0.0

    detection_pattern = {
        **counts,
        "ai_pct": pct(counts["ai"]),
        "human_pct": pct(counts["human"]),
        "uncertain_pct": pct(counts["uncertain"]),
    }

    # ── Appeal rate ───────────────────────────────────────────────────────────
    # Appeals are deduplicated by content_id in case a content_id is appealed
    # more than once (the rate limiter allows up to 3/hour per creator).
    appealed_content_ids = {e["content_id"] for e in appeal_entries}
    appeal_rate = {
        "total_appeals": len(appeal_entries),
        "unique_appealed_content": len(appealed_content_ids),
        "total_decisions": total,
        "rate_pct": (
            round((len(appealed_content_ids) / total) * 100, 2) if total else 0.0
        ),
    }

    # ── Average confidence by attribution ────────────────────────────────────
    avg_confidence = {}
    for label in ("ai", "human", "uncertain"):
        scores = [
            e["confidence"] for e in decision_entries if e.get("attribution") == label
        ]
        avg_confidence[label] = round(sum(scores) / len(scores), 4) if scores else None

    # ── Signal agreement rate ────────────────────────────────────────────────
    # How often do Signal 1 (LLM) and Signal 2 (stylometric) verdicts agree?
    # This is a direct window into how often the disagreement penalty fires.
    agree = 0
    disagree = 0
    for entry in decision_entries:
        signals = entry.get("signals", {})
        llm_verdict = signals.get("llm", {}).get("verdict")
        stylo_verdict = signals.get("stylo", {}).get("verdict")
        if llm_verdict is None or stylo_verdict is None:
            continue
        if llm_verdict == stylo_verdict:
            agree += 1
        else:
            disagree += 1

    signal_total = agree + disagree
    signal_agreement_rate = {
        "agree_count": agree,
        "disagree_count": disagree,
        "rate_pct": round((agree / signal_total) * 100, 2) if signal_total else 0.0,
    }

    return {
        "total_submissions": total,
        "detection_pattern": detection_pattern,
        "appeal_rate": appeal_rate,
        "avg_confidence_by_attribution": avg_confidence,
        "signal_agreement_rate": signal_agreement_rate,
    }


def get_log(limit: int = 50) -> list[dict]:
    """
    Return the most recent *limit* audit log entries as a list of dicts.
    Used by GET /log and for inspection during testing.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(row["raw_entry"]) for row in rows]
    finally:
        conn.close()


def get_entry_by_content_id(content_id: str) -> dict | None:
    """
    Return the most recent decision entry for *content_id*, or None if not found.
    Used by the appeal endpoint to validate the content_id and verify creator_id.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT raw_entry FROM audit_log
            WHERE content_id = ? AND entry_type = 'decision'
            ORDER BY id DESC LIMIT 1
            """,
            (content_id,),
        ).fetchone()
        return json.loads(row["raw_entry"]) if row else None
    finally:
        conn.close()


def log_appeal(
    appeal_id: str,
    content_id: str,
    creator_id: str,
    reasoning: str,
    evidence_url: str | None,
    submitted_at: str,
) -> None:
    """
    Write an appeal entry to the audit log and update the original
    decision's status to 'under_review'.

    Two writes happen:
    1. A new row with entry_type='appeal' capturing the creator's reasoning.
    2. The original decision row's status updated to 'under_review' so the
       full log reflects the current state of the content.
    """
    raw_entry = {
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": submitted_at,
        "entry_type": "appeal",
        "reasoning": reasoning,
        "evidence_url": evidence_url,
        "status": "under_review",
    }

    conn = _get_connection()
    try:
        # Write the appeal entry
        conn.execute(
            """
            INSERT INTO audit_log
                (content_id, creator_id, timestamp, entry_type,
                 attribution, confidence, llm_score, status, raw_entry)
            SELECT
                ?, ?, ?, 'appeal',
                attribution, confidence, llm_score, 'under_review', ?
            FROM audit_log
            WHERE content_id = ? AND entry_type = 'decision'
            ORDER BY id DESC LIMIT 1
            """,
            (content_id, creator_id, submitted_at, json.dumps(raw_entry), content_id),
        )
        # Update the original decision's status in its raw_entry JSON
        row = conn.execute(
            """
            SELECT id, raw_entry FROM audit_log
            WHERE content_id = ? AND entry_type = 'decision'
            ORDER BY id DESC LIMIT 1
            """,
            (content_id,),
        ).fetchone()
        if row:
            updated = json.loads(row["raw_entry"])
            updated["status"] = "under_review"
            conn.execute(
                "UPDATE audit_log SET status = 'under_review', raw_entry = ? WHERE id = ?",
                (json.dumps(updated), row["id"]),
            )
        conn.commit()
    finally:
        conn.close()
