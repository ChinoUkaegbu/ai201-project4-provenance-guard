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

from app.audit.logger import log_decision
from app.extensions import limiter
from app.pipeline.llm_classifier import run_llm_classifier
from app.pipeline.stylometric import run_stylometric
from app.pipeline.fusion import fuse

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

    # ── 4. Signal 2 — stylometric engine (pure Python, no API call) ──────────
    signal_2 = run_stylometric(text)

    # ── 5. Confidence fusion ──────────────────────────────────────────────────
    # Combines both scores using a disagreement-penalised weighted average.
    # See planning.md § 3 for the full algorithm.
    confidence, attribution = fuse(
        llm_score=signal_1["raw_score"],
        stylo_score=signal_2["raw_score"],
    )

    # ── 6. Transparency label ─────────────────────────────────────────────────
    # Three variants per planning.md § 4. Confidence value informs the
    # intensity of the language within each variant.
    if attribution == "ai":
        if confidence >= 0.90:
            label = "Likely AI-Generated (high confidence)"
        else:
            label = "Likely AI-Generated (moderate-high confidence)"
    elif attribution == "human":
        if confidence <= 0.10:
            label = "Likely Written by a Person (high confidence)"
        else:
            label = "Likely Written by a Person (moderate-high confidence)"
    else:
        # Uncertain — check whether signals actively disagreed
        disagreement = abs(signal_1["raw_score"] - signal_2["raw_score"])
        if disagreement > 0.30:
            label = "Authorship Unclear — our two signals gave conflicting results"
        else:
            label = "Authorship Unclear"

    # ── 7. Audit log ──────────────────────────────────────────────────────────
    log_decision(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=confidence,
        llm_score=signal_1["raw_score"],
    )

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
                "signal_2": {
                    "verdict": signal_2["verdict"],
                    "raw_score": signal_2["raw_score"],
                    "sub_scores": signal_2["sub_scores"],
                },
                "status": "classified",
            }
        ),
        200,
    )
