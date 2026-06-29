"""
Transparency Label Generator
-----------------------------
Maps (attribution, confidence, llm_score, stylo_score) to one of three label
variants. Copy is taken verbatim from planning.md § 4.

Variant triggers (from planning.md):
    "high_confidence_ai"    — attribution == "ai"    AND confidence >= 0.80
    "high_confidence_human" — attribution == "human" AND confidence <= 0.20
    "uncertain"             — attribution == "uncertain", OR ai/human but
                              confidence falls outside the high-confidence band

Return shape:
    {
        "variant":           str,        # one of the three above
        "headline":          str,
        "body":              str,
        "confidence_phrase": str | None, # intensity qualifier
        "conflict_note":     str | None  # set when signals actively disagree
    }
"""

# ── Label copy (verbatim from planning.md § 4) ────────────────────────────────

_AI_HEADLINE = "Likely AI-Generated"

_AI_BODY = (
    "Our analysis suggests this content was probably written with AI assistance. "
    "Two independent checks — one looking at writing structure and rhythm, one "
    "assessing overall voice and coherence — both point in the same direction. "
    "This label doesn't mean the content is low quality or that the author did "
    "anything wrong. It's here so you can read with that context in mind."
)

_HUMAN_HEADLINE = "Likely Written by a Person"

_HUMAN_BODY = (
    "Our analysis suggests this content was probably written by a person without "
    "significant AI assistance. Two independent checks found writing patterns — "
    "in both structure and voice — consistent with human authorship. This is a "
    "probabilistic assessment, not a guarantee."
)

_UNCERTAIN_HEADLINE = "Authorship Unclear"

_UNCERTAIN_BODY = (
    "Our analysis wasn't able to reach a confident conclusion about how this "
    "content was written. The signals we use can be inconclusive on mixed or "
    "atypical content — heavily edited AI text, AI-assisted human writing, or "
    "unusually formal human prose can all look similar to our system. We're "
    "showing you this label in the spirit of transparency, not as an accusation."
)

_CONFLICT_NOTE = (
    "Our two detection methods gave conflicting results for this piece, "
    "which is why we're not drawing a firm conclusion."
)


# ── Public function ───────────────────────────────────────────────────────────


def generate_label(
    attribution: str,
    confidence: float,
    llm_score: float,
    stylo_score: float,
) -> dict:
    """
    Generate the transparency label for display to end users.

    Args:
        attribution:  "ai" | "human" | "uncertain" — fused attribution result
        confidence:   fused confidence score 0.0-1.0
        llm_score:    Signal A raw score (for conflict detection)
        stylo_score:  Signal B raw score (for conflict detection)

    Returns:
        {
            "variant":           str,
            "headline":          str,
            "body":              str,
            "confidence_phrase": str | None,
            "conflict_note":     str | None
        }
    """
    disagreement = abs(llm_score - stylo_score)
    conflict_note = _CONFLICT_NOTE if disagreement > 0.30 else None

    # ── Variant 1: High-confidence AI ────────────────────────────────────────
    if attribution == "ai" and confidence >= 0.80:
        if confidence >= 0.90:
            confidence_phrase = "High confidence"
        else:
            confidence_phrase = "Moderate-high confidence"

        return {
            "variant": "high_confidence_ai",
            "headline": _AI_HEADLINE,
            "body": _AI_BODY,
            "confidence_phrase": confidence_phrase,
            "conflict_note": None,  # signals agreed to reach this variant
        }

    # ── Variant 2: High-confidence human ─────────────────────────────────────
    if attribution == "human" and confidence <= 0.20:
        if confidence <= 0.10:
            confidence_phrase = "High confidence"
        else:
            confidence_phrase = "Moderate-high confidence"

        return {
            "variant": "high_confidence_human",
            "headline": _HUMAN_HEADLINE,
            "body": _HUMAN_BODY,
            "confidence_phrase": confidence_phrase,
            "conflict_note": None,
        }

    # ── Variant 3: Uncertain (everything else) ────────────────────────────────
    # Covers: attribution == "uncertain", OR ai/human without enough confidence
    # to reach the high-confidence threshold.
    return {
        "variant": "uncertain",
        "headline": _UNCERTAIN_HEADLINE,
        "body": _UNCERTAIN_BODY,
        "confidence_phrase": None,
        "conflict_note": conflict_note,
    }
