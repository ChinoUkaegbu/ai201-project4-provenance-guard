"""
Confidence Fusion Layer
-----------------------
Combines Signal A (LLM) and Signal B (stylometric) raw scores into a single
calibrated confidence score using a disagreement-penalised weighted average.

Spec (from planning.md § 3):

    WEIGHT_LLM   = 0.60
    WEIGHT_STYLO = 0.40

    weighted_avg = (WEIGHT_LLM * llm_score) + (WEIGHT_STYLO * stylo_score)
    disagreement = abs(llm_score - stylo_score)
    penalty      = disagreement * 0.15

    if weighted_avg > 0.5:  fused = weighted_avg - penalty
    else:                   fused = weighted_avg + penalty

Threshold table (from planning.md § 3):
    0.00 – 0.30  →  "human"
    0.31 – 0.69  →  "uncertain"
    0.70 – 1.00  →  "ai"
"""

WEIGHT_LLM = 0.60
WEIGHT_STYLO = 0.40
DISAGREEMENT_PENALTY_FACTOR = 0.15

THRESHOLD_AI = 0.70
THRESHOLD_HUMAN = 0.30


def fuse(llm_score: float, stylo_score: float) -> tuple[float, str]:
    """
    Fuse two raw signal scores into a single confidence value and label.

    Args:
        llm_score:   0.0 (confident human) → 1.0 (confident AI)
        stylo_score: 0.0 (confident human) → 1.0 (confident AI)

    Returns:
        (fused_confidence, attribution_result)
        where attribution_result is "ai" | "human" | "uncertain"
    """
    # Step 1: weighted average — LLM carries more weight (semantic > structural)
    weighted_avg = (WEIGHT_LLM * llm_score) + (WEIGHT_STYLO * stylo_score)

    # Step 2: disagreement penalty — pull toward 0.5 when signals conflict
    # This makes uncertainty *visible* in the score rather than hiding it.
    # Max possible penalty: abs(1.0 - 0.0) * 0.15 = 0.15
    disagreement = abs(llm_score - stylo_score)
    penalty = disagreement * DISAGREEMENT_PENALTY_FACTOR

    if weighted_avg > 0.5:
        fused = weighted_avg - penalty
    else:
        fused = weighted_avg + penalty

    # Step 3: clamp to [0, 1] and round
    fused = round(max(0.0, min(1.0, fused)), 4)

    # Step 4: apply threshold table from planning.md
    if fused >= THRESHOLD_AI:
        attribution_result = "ai"
    elif fused <= THRESHOLD_HUMAN:
        attribution_result = "human"
    else:
        attribution_result = "uncertain"

    return fused, attribution_result
