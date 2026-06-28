"""
POST /submit
------------
Accepts a JSON body with at minimum a `text` field and a `creator_id` field.
Returns a hardcoded response for now so the route can be verified before
any pipeline logic is added.
"""

from flask import Blueprint, jsonify, request

from app.extensions import limiter

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

    # ── 2. Hardcoded stub response ────────────────────────────────────────────
    # Real pipeline (LLM classifier → stylometric → fusion → label) goes here
    # in the next milestone. Shape matches the final contract so this can be
    # verified end-to-end right now.
    return (
        jsonify(
            {
                "attribution_result": "human",
                "confidence": 0.92,
                "label": {
                    "variant": "high_confidence_human",
                    "headline": "Likely Written by a Person",
                    "body": "This is a hardcoded stub response. Pipeline not yet wired.",
                    "confidence_phrase": "High confidence",
                },
                "signals": {
                    "llm_classifier": None,
                    "stylometric": None,
                },
                "creator_id": creator_id,
                "status": "stub",
            }
        ),
        200,
    )
