"""
POST /appeal/<content_id>
--------------------------
Lets a creator contest a classification.

Request body (JSON):
    {
        "creator_id":    str,           # must match the original submission
        "reasoning":     str,           # max 1000 chars
        "evidence_url":  str | None     # optional supporting link
    }

Response body (JSON):
    {
        "appeal_id":    str,            # uuid4
        "content_id":   str,
        "status":       "under_review",
        "submitted_at": str             # ISO 8601
    }

Error responses:
    400  missing or invalid fields
    403  creator_id does not match original submission
    404  content_id not found in audit log
    429  rate limit exceeded (3 per hour per creator_id)
"""

import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from app.audit.logger import log_appeal, get_entry_by_content_id
from app.extensions import limiter

appeals_bp = Blueprint("appeals", __name__)


@appeals_bp.route("/appeal/<string:content_id>", methods=["POST"])
@limiter.limit(
    "3 per hour",
    key_func=lambda: (request.get_json(silent=True) or {}).get(
        "creator_id", "anonymous"
    ),
)
def appeal(content_id: str):
    # ── 1. Parse & validate request body ─────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    creator_id = body.get("creator_id", "").strip()
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    reasoning = body.get("reasoning", "").strip()
    if not reasoning:
        return jsonify({"error": "Field 'reasoning' is required."}), 400
    if len(reasoning) > 1000:
        return (
            jsonify({"error": "Field 'reasoning' must be 1000 characters or fewer."}),
            400,
        )

    evidence_url = body.get("evidence_url")

    # ── 2. Look up the original decision ─────────────────────────────────────
    original = get_entry_by_content_id(content_id)
    if original is None:
        return jsonify({"error": f"content_id '{content_id}' not found."}), 404

    # ── 3. Verify creator_id matches the original submission ─────────────────
    if original.get("creator_id") != creator_id:
        return (
            jsonify({"error": "creator_id does not match the original submission."}),
            403,
        )

    # ── 4. Write appeal entry + update status to "under_review" ──────────────
    appeal_id = str(uuid.uuid4())
    submitted_at = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )

    log_appeal(
        appeal_id=appeal_id,
        content_id=content_id,
        creator_id=creator_id,
        reasoning=reasoning,
        evidence_url=evidence_url,
        submitted_at=submitted_at,
    )

    # ── 5. Respond ────────────────────────────────────────────────────────────
    return (
        jsonify(
            {
                "appeal_id": appeal_id,
                "content_id": content_id,
                "status": "under_review",
                "submitted_at": submitted_at,
                "message": "Appeals are typically reviewed within 48 hours.",
            }
        ),
        200,
    )
