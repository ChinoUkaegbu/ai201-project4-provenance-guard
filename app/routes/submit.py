"""
POST /submit
------------
Accepts a JSON body with a `text` field and a `creator_id` field.
Runs Signal 1 (LLM classifier) and returns:
  - content_id       : unique ID for this submission (needed by appeals)
  - attribution      : "ai" | "human" from the LLM signal
  - confidence       : raw_score from the LLM signal (placeholder until fusion)
  - label            : placeholder label text until the label generator is built
  - signal_1         : full Signal 1 result for transparency
  - creator_id       : echoed back from the request
"""

import uuid

from flask import Blueprint, jsonify, request

from app.extensions import limiter
from app.pipeline.llm_classifier import run_llm_classifier

submit_bp = Blueprint("submit", __name__)


@submit_bp.route("/submit", methods=["POST"])
@limiter.limit("20 per minute")
@limiter.limit("500 per day")
def submit():
    # ── 1. Parse & validate ───────────────────────────────────────────────────
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    text = body.get("text", "").strip()
    if not text:
        return (
            jsonify({"error": "Field 'text' is required and must not be empty."}),
            400,
        )

    creator_id = body.get("creator_id", "").strip()
    if not creator_id:
        return (
            jsonify({"error": "Field 'creator_id' is required and must not be empty."}),
            400,
        )

    # ── 2. Generate content_id ────────────────────────────────────────────────
    # This ID ties together the submission, the audit log entry, and any
    # future appeal. It must be returned in every response.
    content_id = str(uuid.uuid4())

    # ── 3. Signal 1 — LLM classifier ─────────────────────────────────────────
    try:
        signal_1 = run_llm_classifier(text)
    except RuntimeError as exc:
        return (
            jsonify(
                {
                    "error": "Signal 1 (LLM classifier) failed.",
                    "detail": str(exc),
                }
            ),
            500,
        )

    # ── 4. Build response ─────────────────────────────────────────────────────
    # confidence and label are placeholders until Signal 2 + fusion are wired in.
    attribution = signal_1["verdict"]
    confidence = signal_1["raw_score"]  # placeholder: will become fused score

    # Placeholder label — will be replaced by the label generator
    if confidence >= 0.70:
        label = "Likely AI-Generated (placeholder)"
    elif confidence <= 0.30:
        label = "Likely Written by a Person (placeholder)"
    else:
        label = "Authorship Unclear (placeholder)"

    return (
        jsonify(
            {
                "content_id": content_id,
                "creator_id": creator_id,
                "attribution": attribution,
                "confidence": confidence,
                "label": label,
                "signal_1": {
                    "verdict": signal_1["verdict"],
                    "raw_score": signal_1["raw_score"],
                    "reasoning_excerpt": signal_1["reasoning_excerpt"],
                },
                "status": "classified",
            }
        ),
        200,
    )
