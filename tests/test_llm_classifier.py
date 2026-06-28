"""
test_llm_classifier.py
----------------------
Run this directly to test the LLM classifier signal in isolation,
before it is wired into the POST /submit endpoint.

Usage:
    cd ai201-project4-provenance-guard
    python tests/test_llm_classifier.py

Requires GROQ_API_KEY in your .env file.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from app.pipeline.llm_classifier import run_llm_classifier, LLMSignalResult

# ── Test inputs ───────────────────────────────────────────────────────────────
# Three texts chosen to stress-test distinct scenarios:
#   1. A clearly AI-written passage  → expect high raw_score, verdict "ai"
#   2. A clearly human-written piece → expect low raw_score,  verdict "human"
#   3. An ambiguous case             → expect mid-range score, either verdict

TEST_CASES = [
    {
        "label": "Likely AI — polished blog intro",
        "text": (
            "Artificial intelligence is transforming the way we approach "
            "problem-solving across industries. From healthcare to finance, "
            "the applications of machine learning are vast and growing. "
            "In this article, we will explore the key benefits of AI adoption "
            "and discuss how organisations can leverage these technologies "
            "to drive innovation and improve operational efficiency."
        ),
        "expect_verdict": "ai",
    },
    {
        "label": "Likely Human — personal journal entry",
        "text": (
            "I don't know why I even started writing this. It's 2am and I "
            "can't sleep again — keep thinking about what Mara said at dinner, "
            "the part where she went quiet mid-sentence and just looked at her "
            "hands. I've replayed it maybe forty times. Probably means nothing. "
            "Probably means everything. I should text her but I won't."
        ),
        "expect_verdict": "human",
    },
    {
        "label": "Ambiguous — simple descriptive prose",
        "text": (
            "The park was quiet in the early morning. A few joggers passed "
            "along the main path. The trees had started to turn, orange and "
            "yellow against a grey sky. Someone had left a coffee cup on the "
            "bench by the pond. A duck investigated it briefly, then moved on."
        ),
        "expect_verdict": None,  # no strong expectation — inspect the score
    },
]


def run_tests():
    print("=" * 60)
    print("LLM Classifier — standalone signal test")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {case['label']}")
        print(f"  Text preview: {case['text'][:80].strip()}...")

        try:
            result: LLMSignalResult = run_llm_classifier(case["text"])
        except RuntimeError as exc:
            print(f"  FAIL — RuntimeError: {exc}")
            failed += 1
            continue

        # ── Print full result ─────────────────────────────────────────────
        print(f"  verdict          : {result['verdict']}")
        print(f"  raw_score        : {result['raw_score']}")
        print(f"  reasoning_excerpt: {result['reasoning_excerpt']}")

        # ── Structural checks (always) ────────────────────────────────────
        assert (
            result["signal"] == "llm_classifier"
        ), f"Expected signal='llm_classifier', got {result['signal']!r}"
        assert result["verdict"] in (
            "ai",
            "human",
        ), f"verdict must be 'ai' or 'human', got {result['verdict']!r}"
        assert isinstance(
            result["raw_score"], float
        ), f"raw_score must be a float, got {type(result['raw_score'])}"
        assert (
            0.0 <= result["raw_score"] <= 1.0
        ), f"raw_score must be in [0,1], got {result['raw_score']}"
        assert (
            isinstance(result["reasoning_excerpt"], str) and result["reasoning_excerpt"]
        ), "reasoning_excerpt must be a non-empty string"

        # ── Verdict check (where we have an expectation) ──────────────────
        if case["expect_verdict"] is not None:
            if result["verdict"] == case["expect_verdict"]:
                print(f"  PASS — verdict matches expected '{case['expect_verdict']}'")
                passed += 1
            else:
                print(
                    f"  WARN — expected '{case['expect_verdict']}', "
                    f"got '{result['verdict']}' (score {result['raw_score']}). "
                    f"Not a hard failure — inspect reasoning above."
                )
                # Treat as pass if score is in the uncertain zone (0.35–0.65)
                if 0.35 <= result["raw_score"] <= 0.65:
                    print("       Score is in uncertain zone — acceptable.")
                    passed += 1
                else:
                    failed += 1
        else:
            print(f"  PASS — no verdict expectation, structure checks passed.")
            passed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
