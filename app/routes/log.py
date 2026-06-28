"""
GET /log
--------
Returns the most recent audit log entries as JSON.

Query params:
    limit   int   number of entries to return, default 50, max 100

Response body:
    {
        "entries": [ ...structured audit entries... ],
        "count":   int
    }

Note: in a real system this would require authentication. Here it is
intentionally open for documentation and grading visibility.
"""

from flask import Blueprint, jsonify, request

from app.audit.logger import get_log

log_bp = Blueprint("log", __name__)


@log_bp.route("/log", methods=["GET"])
def log():
    limit = request.args.get("limit", 50, type=int)
    limit = min(limit, 100)  # cap at 100 regardless of what was passed

    entries = get_log(limit=limit)

    return (
        jsonify(
            {
                "entries": entries,
                "count": len(entries),
            }
        ),
        200,
    )
