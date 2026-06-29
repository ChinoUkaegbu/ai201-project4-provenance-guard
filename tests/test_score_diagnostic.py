"""
test_score_diagnostic.py
-------------------
Runs all four test inputs through Signal B (stylometric) and the fusion layer,
then prints a full breakdown so you can check whether the scores match your
intuition.

Signal A (LLM) requires a live Groq API key. If GROQ_API_KEY is set in your
.env, you'll see real LLM scores. If not, the script uses the hardcoded
approximate scores from the previous milestone test run so you can still
inspect the stylometric and fusion results.

Usage:
    cd ai201-project4-provenance-guard
    python tests/test_score_diagnostic.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from app.pipeline.stylometric import run_stylometric
from app.pipeline.fusion import fuse

# ── Test inputs ───────────────────────────────────────────────────────────────

TEXTS = [
    {
        "label": "Clearly AI — formal paradigm-shift prose",
        "expect": "high score (AI)",
        "text": (
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that while the benefits "
            "of AI are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment."
        ),
    },
    {
        "label": "Clearly Human — casual ramen review",
        "expect": "low score (human)",
        "text": (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after. my friend got "
            "the spicy version and said it was better. probably won't go back "
            "unless someone drags me there"
        ),
    },
    {
        "label": "Borderline — formal human academic writing",
        "expect": "mid-high (stylometrics may flag as AI)",
        "text": (
            "The relationship between monetary policy and asset price inflation "
            "has been extensively studied in the literature. Central banks face "
            "a fundamental tension between their mandate for price stability and "
            "the unintended consequences of prolonged low interest rates on "
            "equity and real estate valuations."
        ),
    },
    {
        "label": "Borderline — lightly edited AI output",
        "expect": "mid-range",
        "text": (
            "I've been thinking a lot about remote work lately. There are genuine "
            "tradeoffs — flexibility and no commute on one side, isolation and "
            "blurred work-life boundaries on the other. Studies show productivity "
            "varies widely by individual and role type."
        ),
    },
]

# ── Fallback LLM scores (used when no API key is present) ────────────────────
# These are plausible estimates based on what the LLM returned in earlier tests.
# Replace them with real scores once you've run the live test.
FALLBACK_LLM_SCORES = {
    "Clearly AI — formal paradigm-shift prose": 0.80,
    "Clearly Human — casual ramen review": 0.20,
    "Borderline — formal human academic writing": 0.70,
    "Borderline — lightly edited AI output": 0.70,
}


def get_llm_score(text: str, label: str) -> tuple[float, str, bool]:
    """
    Try to get a real LLM score. Falls back to the hardcoded estimate
    if GROQ_API_KEY is not set. Returns (score, reasoning, is_live).
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        fallback = FALLBACK_LLM_SCORES[label]
        return fallback, "(fallback — no API key)", False

    try:
        from app.pipeline.llm_classifier import run_llm_classifier

        result = run_llm_classifier(text)
        return result["raw_score"], result["reasoning_excerpt"], True
    except Exception as exc:
        fallback = FALLBACK_LLM_SCORES[label]
        return fallback, f"(fallback — API error: {exc})", False


# ── Bar chart helper ──────────────────────────────────────────────────────────


def bar(score: float, width: int = 30) -> str:
    """Render a simple ASCII bar for a 0–1 score."""
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.3f}"


# ── Main ──────────────────────────────────────────────────────────────────────


def run():
    print("=" * 70)
    print("  Provenance Guard — Score Diagnostic")
    print("=" * 70)

    api_key_present = bool(os.environ.get("GROQ_API_KEY", ""))
    if api_key_present:
        print("  Signal A: LIVE  (Groq API key found)")
    else:
        print(
            "  Signal A: FALLBACK  (no GROQ_API_KEY — set it in .env for live scores)"
        )
    print("  Signal B: LIVE  (pure Python, no key needed)")
    print()

    for i, case in enumerate(TEXTS, 1):
        print(f"{'─' * 70}")
        print(f"  [{i}] {case['label']}")
        print(f"       Expected: {case['expect']}")
        print()

        # ── Signal A ─────────────────────────────────────────────────────────
        llm_score, reasoning, is_live = get_llm_score(case["text"], case["label"])
        live_tag = "live" if is_live else "fallback"
        print(f"  SIGNAL A — LLM classifier ({live_tag})")
        print(f"    score     {bar(llm_score)}")
        print(f"    reasoning {reasoning[:80]}")
        print()

        # ── Signal B ─────────────────────────────────────────────────────────
        stylo = run_stylometric(case["text"])
        print(f"  SIGNAL B — Stylometric")
        print(f"    score     {bar(stylo['raw_score'])}")
        ss = stylo["sub_scores"]
        print(
            f"    ├ sentence_length_variance   {bar(ss['sentence_length_variance'], 20)}"
        )
        print(f"    ├ type_token_ratio            {bar(ss['type_token_ratio'], 20)}")
        print(
            f"    ├ exotic_punctuation_density  {bar(ss['exotic_punctuation_density'], 20)}"
        )
        print(f"    └ burstiness_index            {bar(ss['burstiness_index'], 20)}")
        print()

        # ── Fusion ───────────────────────────────────────────────────────────
        confidence, attribution = fuse(llm_score, stylo["raw_score"])
        disagreement = abs(llm_score - stylo["raw_score"])

        print(f"  FUSION")
        print(f"    Signal A ({0.60:.0%} weight): {llm_score:.3f}")
        print(f"    Signal B ({0.40:.0%} weight): {stylo['raw_score']:.3f}")
        print(
            f"    Disagreement:          {disagreement:.3f}  →  penalty {disagreement * 0.15:.3f}"
        )
        print(f"    Fused confidence: {bar(confidence)}")
        print(f"    Attribution:      {attribution.upper()}")
        print()

    print("=" * 70)
    print()
    print("  SUMMARY")
    print(f"  {'Input':<45} {'LLM':>6} {'Stylo':>6} {'Fused':>6}  Label")
    print(f"  {'─' * 45} {'─' * 6} {'─' * 6} {'─' * 6}  {'─' * 12}")
    for case in TEXTS:
        llm_score, _, _ = get_llm_score(case["text"], case["label"])
        stylo = run_stylometric(case["text"])
        confidence, attribution = fuse(llm_score, stylo["raw_score"])
        print(
            f"  {case['label']:<45} {llm_score:>6.3f} {stylo['raw_score']:>6.3f} {confidence:>6.3f}  {attribution}"
        )
    print()
    print("  Thresholds:  >= 0.70 → ai  |  0.31–0.69 → uncertain  |  <= 0.30 → human")
    print("=" * 70)


if __name__ == "__main__":
    run()
