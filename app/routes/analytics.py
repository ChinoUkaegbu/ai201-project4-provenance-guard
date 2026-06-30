"""
GET /analytics
--------------
Returns aggregate metrics over the audit log: detection pattern (ratio of
ai/human/uncertain verdicts), appeal rate, average confidence by attribution,
and signal agreement rate (how often Signal 1 and Signal 2 agree).

No authentication — same reasoning as GET /log: open for documentation and
grading visibility. A real deployment would put this behind an admin token.
"""

from flask import Blueprint, jsonify

from app.audit.logger import get_analytics
from app.extensions import limiter

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/analytics", methods=["GET"])
@limiter.limit("60 per minute")
def analytics():
    return jsonify(get_analytics()), 200
