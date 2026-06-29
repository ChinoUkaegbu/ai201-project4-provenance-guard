"""
Signal B — Stylometric Heuristic Engine
-----------------------------------------
Pure Python. No model dependency. Measures four structural-statistical
properties that differ between human and AI writing.

HOW EACH SUB-SIGNAL IS SCORED (0.0 = human, 1.0 = AI):

1. Sentence Length Variance
   AI writing clusters near a comfortable mean — σ is low.
   Human writing swings: long when exploring, short for impact.
   We measure std dev of sentence word counts, then invert-normalise:
   low variance → high (AI) score.
   Reference range: σ=0 (perfectly uniform) → σ=20 (very bursty).

2. Type-Token Ratio (TTR)
   TTR = unique words / total words.
   AI writing tends toward middling TTR — broad but not weird.
   Extremely high TTR (every word unique) or extremely low (lots of
   repetition) both suggest human intentionality.
   We score distance from the AI "comfortable band" of 0.55–0.75:
   inside the band → high (AI) score; outside → lower score.

3. Exotic Punctuation Density
   Em-dashes, ellipses, semicolons, parentheticals per 100 words.
   Humans punctuate for rhythm. LLMs default to commas and periods.
   High exotic punctuation → low (human) score.
   Reference range: 0 (none) → 5+ per 100 words (very human).

4. Burstiness Index (Coefficient of Variation)
   CV = std_dev / mean of sentence lengths.
   AI prose: low CV (uniform). Human prose: high CV (bursty).
   High CV → low (human) score.
   Reference range: CV=0 (uniform) → CV=1.5 (very bursty).
"""

import re
import math

# ── Reference ranges for normalisation ───────────────────────────────────────
# These are calibrated against a small reference set of known human and AI
# texts. Tweak as you gather more data.

_VARIANCE_MAX = 20.0  # std dev above this → very human-like
_BURSTINESS_MAX = 1.5  # CV above this → very human-like
_EXOTIC_MAX = 5.0  # exotic chars per 100 words above this → very human
_TTR_AI_LOW = 0.55  # TTR band that is typical of AI writing
_TTR_AI_HIGH = 0.75


# ── Helpers ───────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Sub-signal scorers ────────────────────────────────────────────────────────


def _score_sentence_length_variance(sentences: list[str]) -> float:
    """
    Low variance → high AI score.
    Returns 0.0 (very bursty = human) → 1.0 (very uniform = AI).
    """
    if len(sentences) < 2:
        return 0.5  # not enough data — return neutral

    lengths = [_word_count(s) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)

    # Invert: high std_dev → low AI score
    ai_score = 1.0 - _clamp(std_dev / _VARIANCE_MAX)
    return round(ai_score, 4)


def _score_type_token_ratio(text: str) -> float:
    """
    TTR inside the AI comfort band (0.55–0.75) → high AI score.
    TTR outside that band (very high or very low) → lower AI score.
    Returns 0.0 (clearly non-AI TTR) → 1.0 (TTR in AI comfort band).
    """
    words = text.lower().split()
    if len(words) < 10:
        return 0.5

    ttr = len(set(words)) / len(words)

    # Distance from centre of AI band (0.65)
    band_centre = (_TTR_AI_LOW + _TTR_AI_HIGH) / 2  # 0.65
    band_half_width = (_TTR_AI_HIGH - _TTR_AI_LOW) / 2  # 0.10

    distance = abs(ttr - band_centre)
    # If inside the band, distance < band_half_width → high AI score
    ai_score = _clamp(1.0 - (distance / (band_half_width * 3)))
    return round(ai_score, 4)


def _score_exotic_punctuation(text: str, word_count: int) -> float:
    """
    High exotic punctuation density → low AI score (human-like).
    Returns 0.0 (very exotic = human) → 1.0 (plain punctuation = AI).
    """
    if word_count == 0:
        return 0.5

    # Count exotic punctuation marks
    exotic_chars = re.findall(r"[—–…;()]", text)
    density_per_100 = (len(exotic_chars) / word_count) * 100

    # Invert: high density → low AI score
    ai_score = 1.0 - _clamp(density_per_100 / _EXOTIC_MAX)
    return round(ai_score, 4)


def _score_burstiness(sentences: list[str]) -> float:
    """
    High coefficient of variation (bursty) → low AI score (human-like).
    Returns 0.0 (very bursty = human) → 1.0 (uniform = AI).
    """
    if len(sentences) < 2:
        return 0.5

    lengths = [_word_count(s) for s in sentences]
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.5

    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean  # coefficient of variation

    # Invert: high CV → low AI score
    ai_score = 1.0 - _clamp(cv / _BURSTINESS_MAX)
    return round(ai_score, 4)


# ── Public function ───────────────────────────────────────────────────────────


def run_stylometric(text: str) -> dict:
    """
    Run all four stylometric sub-signals and combine into a single score.

    Args:
        text: The raw text to analyse.

    Returns:
        {
            "signal":    "stylometric",
            "verdict":   "ai" | "human",
            "raw_score": float,   # 0.0 = confident human, 1.0 = confident AI
            "sub_scores": {
                "sentence_length_variance":   float,
                "type_token_ratio":           float,
                "exotic_punctuation_density": float,
                "burstiness_index":           float
            }
        }
    """
    sentences = _split_sentences(text)
    wc = _word_count(text)

    sub_scores = {
        "sentence_length_variance": _score_sentence_length_variance(sentences),
        "type_token_ratio": _score_type_token_ratio(text),
        "exotic_punctuation_density": _score_exotic_punctuation(text, wc),
        "burstiness_index": _score_burstiness(sentences),
    }

    # Simple average of the four sub-scores
    raw_score = round(sum(sub_scores.values()) / len(sub_scores), 4)
    verdict = "ai" if raw_score >= 0.5 else "human"

    return {
        "signal": "stylometric",
        "verdict": verdict,
        "raw_score": raw_score,
        "sub_scores": sub_scores,
    }
