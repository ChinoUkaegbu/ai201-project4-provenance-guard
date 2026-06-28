import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ── Confidence fusion weights ─────────────────────────────────────────────────
# LLM signal carries more weight: it captures semantics, not just structure.
WEIGHT_LLM: float = 0.60
WEIGHT_STYLO: float = 0.40

# Max pull toward 0.5 when signals actively disagree (see fusion.py).
DISAGREEMENT_PENALTY_FACTOR: float = 0.15

# ── Classification thresholds ─────────────────────────────────────────────────
# Fused score >= THRESHOLD_AI   → label "ai"
# Fused score <= THRESHOLD_HUMAN → label "human"
# Anything in between           → label "uncertain"
THRESHOLD_AI: float = 0.70
THRESHOLD_HUMAN: float = 0.30

# Texts shorter than this word count get a mandatory "uncertain" label
# regardless of signal scores (stylometrics are unreliable on short text).
MIN_WORDS_FOR_CONFIDENT_LABEL: int = 100

# ── Rate limiting ─────────────────────────────────────────────────────────────
# POST /submit  — 20/min keeps us inside free-tier Groq limits
RATE_LIMIT_SUBMIT_PER_MINUTE: str = "20 per minute"
RATE_LIMIT_SUBMIT_PER_DAY: str = "500 per day"
# POST /appeal  — 3/hour: appeals are a human action, not a bulk operation
RATE_LIMIT_APPEAL_PER_HOUR: str = "3 per hour"
# GET /log      — admin endpoint, generous but bounded
RATE_LIMIT_LOG_PER_MINUTE: str = "60 per minute"
