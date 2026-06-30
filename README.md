# Provenance Guard - Project 4

A backend system that classifies submitted text content for likely AI vs. human authorship, scores confidence in that classification, surfaces a plain-language transparency label, and handles creator appeals. Built to be pluggable into any creative-sharing platform.

---

## Stack

| Component          | Tool                                 |
| ------------------ | ------------------------------------ |
| API framework      | Flask                                |
| Detection signal 1 | Groq (`llama-3.3-70b-versatile`)     |
| Detection signal 2 | Stylometric heuristics (pure Python) |
| Rate limiting      | Flask-Limiter                        |
| Audit log          | SQLite                               |

---

## Setup

```
Add your Groq API key to .env
pip install -r requirements.txt
python -m app.main
```

The server runs at `http://127.0.0.1:5000`. The audit log database (`audit_log.db`) is created automatically on first use.

---

## Endpoints

### `POST /submit`

```json
{
  "text": "the content to analyse",
  "creator_id": "platform-user-id"
}
```

Returns the attribution result, fused confidence score, transparency label, and a breakdown of both signals. Rate limited to 20/minute and 500/day.

### `POST /appeal/<content_id>`

```json
{
  "creator_id": "must match the original submission",
  "reasoning": "free text, max 1000 chars",
  "evidence_url": "optional"
}
```

Updates the content's status to `under_review`, logs the appeal, returns an `appeal_id`. Rate limited to 3/hour per `creator_id`.

### `GET /log`

Returns the most recent audit log entries. Rate limited to 60/minute. No authentication — intentionally open for grading/documentation visibility (a real deployment would require an admin token here).

---

## Detection Signals — Reasoning

I needed two signals that were genuinely independent — not two flavors of the same approach — so that disagreement between them would carry information rather than just being noise.

**Signal 1 — LLM semantic classifier (Groq).** I chose this because it can see something no statistical method can: holistic coherence. A model reading a paragraph can sense whether the voice feels like a person navigating an idea in real time, or like the smooth, risk-averse completion pattern of a language model. This is the signal that should catch content a casual reader would also instinctively flag, because it's evaluating the same thing a human reader notices — voice.

The tradeoff is that this signal is opaque and non-deterministic. I don't fully know what features in the text the model is keying on, and the same text can score slightly differently across calls. It's also vulnerable to adversarial prompting — text written specifically to read as human will fool this signal more easily than it fools a statistical one, because both are optimizing for the same thing: smooth, human-feeling prose.

**Signal 2 — stylometric heuristics (pure Python).** I chose this specifically because it's the opposite of Signal 1 in every way that matters: deterministic, inspectable, and structural rather than semantic. It measures four things — sentence length variance, type-token ratio, exotic punctuation density, and burstiness — all chosen because they capture _regularity_, which is a side effect of how LLMs are trained (toward smooth, readable, consistent output) rather than something an LLM is deliberately optimizing to fake.

The tradeoff is that this signal has no concept of meaning. It can't tell the difference between "this text is uniform because an LLM wrote it" and "this text is uniform because it's a formal academic abstract" or "this text is uniform because it's a clean, simple short poem." It is blind to exactly the kind of content a human reader would recognize as obviously human at a glance.

**Why combine them with a disagreement penalty rather than a simple average.** A simple average of two confident-but-opposite scores (say, 0.85 and 0.15) would land at 0.50 — which _looks_ like honest uncertainty but is actually hiding a real disagreement between two different kinds of evidence. I wanted the system to be able to say "I don't know" for two different reasons: because the evidence is genuinely weak (`both signals near 0.5`), or because the evidence is _strong but contradictory_ (`one signal near 0, one near 1`). The disagreement penalty makes both of these visible in the score and the label, rather than collapsing them into the same flat 0.5.

**What I'd change for a real deployment.** I'd want to calibrate the fusion weights and thresholds against a real labeled dataset rather than the small set of test inputs I worked with here. Right now the 0.60/0.40 weighting and the 0.70/0.30 thresholds are reasoned defaults, not empirically fitted values. I'd also want per-genre calibration — the stylometric signal's blind spots (poetry, academic writing, casual short-form text) are predictable enough that a genre flag from the platform could let the system down-weight Signal 2 in cases where it's known to be unreliable.

---

## Confidence Scoring — Reasoning

A binary label throws away information the system actually has. Two pieces of content that are both classified "AI" can have very different amounts of supporting evidence behind that classification, and a creator contesting the more uncertain one has a legitimate complaint that a binary system can't represent.

The fusion formula:

```
weighted_avg = (0.60 × llm_score) + (0.40 × stylo_score)
disagreement = abs(llm_score − stylo_score)
penalty = disagreement × 0.15

fused = weighted_avg − penalty   (if weighted_avg > 0.5)
fused = weighted_avg + penalty   (if weighted_avg ≤ 0.5)
```

Thresholds: ≥0.70 → `ai`, ≤0.30 → `human`, in between → `uncertain`.

**How I tested whether the scores are meaningful, not just plausible-looking.** I ran the same set of test texts through both signals independently and recorded where they agreed and where they diverged (see the example pairs below). The test that mattered most wasn't "does a clearly-AI text score high" — that's the easy case. It was checking whether the system's _uncertainty_ was honest: does a text that gets contradictory signals from the two detectors actually land in the uncertain band, or does it get rounded toward whichever signal happened to be stronger? Verifying the disagreement penalty pulls the score toward 0.5 (rather than just averaging) was the key check here.

### Example: high-confidence vs. lower-confidence case

**High-confidence case**

Text:

> "In today's fast-paced, ever-evolving digital landscape, it has become increasingly crucial for businesses to adapt and grow. Whether you are navigating the intricate realm of modern commerce or looking to streamline your daily operations, the power of innovation cannot be overstated. By harnessing transformative technologies, you can unlock your true potential and propel your organization to the forefront."

| Field                            | Value                                             |
| -------------------------------- | ------------------------------------------------- |
| Signal 1 (LLM) raw_score         | 0.90                                              |
| Signal 2 (stylometric) raw_score | 0.768                                             |
| Fused confidence                 | **0.8274**                                        |
| Attribution                      | `ai`                                              |
| Label variant                    | `high_confidence_ai` — "Moderate-high confidence" |

**Lower-confidence case**

Text:

> "Hope you all weathered 2020 as best you could! I know the world threw a lot of crap our way this year — way too much, honestly — but from how I've seen my friends and family respond... I'm proud. Genuinely proud of what we've accomplished despite the unexpected speed bumps in our journey. This comic is the first in a series; the rest will appear in their own publication on Medium — A Dog Named Karma. This one's in Tech Doodles, just so you know I'm working on something new — I think you'll like it too."

| Field                            | Value                                                  |
| -------------------------------- | ------------------------------------------------------ |
| Signal 1 (LLM) raw_score         | 0.12 (verdict: `human`)                                |
| Signal 2 (stylometric) raw_score | 0.6389 (verdict: `ai`)                                 |
| Fused confidence                 | **0.4054**                                             |
| Attribution                      | `uncertain`                                            |
| Label variant                    | `uncertain` — "Authorship Unclear", with conflict note |

**Analysis.** This pair shows two different mechanisms producing uncertainty, not just two different magnitudes. The high-confidence case had both signals agreeing in direction and firing strongly — that's clean, low-conflict evidence. This case is the opposite: the LLM is fairly confident this is human-written (0.12), explicitly citing "colloquial expressions" and "self-referential asides" — exactly the personal, casual voice a human reader would also notice. But the stylometric signal scores it 0.6389 toward AI, driven mainly by `exotic_punctuation_density` (0.7959) and `type_token_ratio` (0.5816) landing in ranges the heuristic treats as AI-typical, despite the text actually containing several em-dashes, an ellipsis, and a semicolon — punctuation a human writer reaches for naturally. The raw disagreement between the two signals here is `|0.12 − 0.6389| = 0.52`, large enough to trigger the conflict note in the label (`disagreement > 0.30`), which is why this response — unlike the remote-work example — explicitly tells the reader the two methods disagreed.

A naive average of 0.12 and 0.6389 would land at roughly 0.38 — close to what we got (0.4054), since the disagreement penalty here is modest (0.52 × 0.15 ≈ 0.078) given the signals aren't at the absolute extremes. But the more important point isn't the arithmetic, it's that the system surfaces _why_ it's uncertain. A reader seeing this label sees an explicit statement that the two detection methods disagreed, which is meaningfully different information than "the evidence was weak on both sides" — and it's the scenario `planning.md`'s Edge Case 4 (adversarial or atypical human writing) anticipated directly.

_Note: this text was deliberately revised through several iterations specifically to test and demonstrate the stylometric signal's punctuation blind spot — it wasn't an organic, unmodified submission. See the AI Usage section for how that iteration happened._

---

## Transparency Label — All Three Variants

### Variant 1 — High-Confidence AI

_Trigger: `attribution == "ai"` and `confidence ≥ 0.80`_

**Headline:** `Likely AI-Generated`

**Body:**

> Our analysis suggests this content was probably written with AI assistance. Two independent checks — one looking at writing structure and rhythm, one assessing overall voice and coherence — both point in the same direction. This label doesn't mean the content is low quality or that the author did anything wrong. It's here so you can read with that context in mind.

**Confidence phrase:** `High confidence` (≥0.90) or `Moderate-high confidence` (0.80–0.89)

---

### Variant 2 — High-Confidence Human

_Trigger: `attribution == "human"` and `confidence ≤ 0.20`_

**Headline:** `Likely Written by a Person`

**Body:**

> Our analysis suggests this content was probably written by a person without significant AI assistance. Two independent checks found writing patterns — in both structure and voice — consistent with human authorship. This is a probabilistic assessment, not a guarantee.

**Confidence phrase:** `High confidence` (≤0.10) or `Moderate-high confidence` (0.11–0.20)

---

### Variant 3 — Uncertain

_Trigger: everything else — including `attribution == "uncertain"`, or `ai`/`human` results that don't clear the high-confidence threshold_

**Headline:** `Authorship Unclear`

**Body:**

> Our analysis wasn't able to reach a confident conclusion about how this content was written. The signals we use can be inconclusive on mixed or atypical content — heavily edited AI text, AI-assisted human writing, or unusually formal human prose can all look similar to our system. We're showing you this label in the spirit of transparency, not as an accusation.

**Additional note (when signals disagree by more than 0.30):**

> Our two detection methods gave conflicting results for this piece, which is why we're not drawing a firm conclusion.

---

## Audit Log

Every decision is written to `audit_log.db` as a structured JSON entry, retrievable via `GET /log`. Each entry contains the timestamp, `content_id`, `creator_id`, attribution, fused confidence, both individual signal scores and verdicts, and a status field (`classified` or `under_review`). Appeals are written as separate linked entries containing the creator's reasoning, and they update the original decision's status.

**Sample entries from `GET /log`** (4 most recent, showing the full decision → appeal → status-change lifecycle):

```json
[
  {
    "appeal_id": "ef1a3d52-1281-4fca-867c-2d8f86083c68",
    "content_id": "5d988396-3e32-4a47-8b7e-6464c3f737db",
    "creator_id": "demo-user-3",
    "entry_type": "appeal",
    "evidence_url": null,
    "reasoning": "This was a personal observation I wrote myself while sitting in the park.",
    "status": "under_review",
    "timestamp": "2026-06-30T08:30:43.554Z"
  },
  {
    "attribution": "ai",
    "confidence": 0.7133,
    "content_id": "5d988396-3e32-4a47-8b7e-6464c3f737db",
    "creator_id": "demo-user-3",
    "entry_type": "decision",
    "signals": {
      "llm": { "score": 0.7, "verdict": "ai" },
      "stylo": { "score": 0.7531, "verdict": "ai" }
    },
    "status": "under_review",
    "timestamp": "2026-06-30T08:29:30.424Z"
  },
  {
    "attribution": "uncertain",
    "confidence": 0.342,
    "content_id": "3c2ef3a4-4c2f-4b37-be7e-7442681d40e3",
    "creator_id": "demo-user-2",
    "entry_type": "decision",
    "signals": {
      "llm": { "score": 0.12, "verdict": "human" },
      "stylo": { "score": 0.5237, "verdict": "ai" }
    },
    "status": "classified",
    "timestamp": "2026-06-30T08:28:20.211Z"
  },
  {
    "attribution": "ai",
    "confidence": 0.7308,
    "content_id": "ae2a3bf7-f5ad-4d0e-831e-f74fa55bc6f3",
    "creator_id": "demo-user-1",
    "entry_type": "decision",
    "signals": {
      "llm": { "score": 0.8, "verdict": "ai" },
      "stylo": { "score": 0.6741, "verdict": "ai" }
    },
    "status": "classified",
    "timestamp": "2026-06-30T07:44:09.317Z"
  }
]
```

Note the first two entries from the top: the same `content_id` (`5d988396-...`) appears twice — once as the original `decision` entry (now showing `status: "under_review"`, updated after the appeal) and once as the linked `appeal` entry containing the creator's reasoning. This demonstrates the full lifecycle: a content piece gets classified, a creator contests it, and the log preserves both the original decision and the contest without overwriting anything.

---

## Analytics Dashboard

`GET /analytics` returns aggregate metrics computed over the audit log, giving a platform operator a sense of how the system is behaving in production rather than just inspecting individual decisions. No new signals or storage were needed — every metric is derived from data already written by `POST /submit` and `POST /appeal`.

**Metrics returned:**

1. **Detection pattern** — counts and percentages of `ai` / `human` / `uncertain` verdicts across all decisions. This is the headline number a platform would want: what fraction of submitted content is the system actually flagging.
2. **Appeal rate** — the percentage of decisions that have been contested, deduplicated by `content_id` (a single piece of content can technically be appealed more than once within the rate limit, but it should only count once toward the rate). A high appeal rate on a particular attribution bucket would be a signal that the system's thresholds or copy need revisiting.
3. **Average confidence by attribution** — the mean fused confidence score within each verdict bucket. This distinguishes "the system says AI a lot, and is usually very sure" from "the system says AI a lot, but usually only just barely" — two very different operational pictures that the raw detection pattern alone can't show.
4. **Signal agreement rate** _(the additional metric)_ — the percentage of decisions where Signal 1 (LLM) and Signal 2 (stylometric) reached the same verdict. This was chosen specifically because it gives direct visibility into how often the disagreement penalty in the fusion layer is actually doing work. A low agreement rate would suggest the two signals are picking up on different things often enough that the `uncertain` label is earning its keep; a very high agreement rate might suggest one signal is redundant or the test content skews toward easy cases.
   **Example response shape:**

```json
{
  "total_submissions": 12,
  "detection_pattern": {
    "ai": 5,
    "human": 3,
    "uncertain": 4,
    "ai_pct": 41.67,
    "human_pct": 25.0,
    "uncertain_pct": 33.33
  },
  "appeal_rate": {
    "total_appeals": 2,
    "unique_appealed_content": 2,
    "total_decisions": 12,
    "rate_pct": 16.67
  },
  "avg_confidence_by_attribution": {
    "ai": 0.8124,
    "human": 0.1432,
    "uncertain": 0.4983
  },
  "signal_agreement_rate": {
    "agree_count": 9,
    "disagree_count": 3,
    "rate_pct": 75.0
  }
}
```

Rate limited to 60/minute, same as `GET /log`, and intentionally open without authentication for the same reason — this is a grading/demo convenience, not a production-ready access pattern. A real deployment would put this behind an admin token.

---

## Rate Limiting

### Limits

| Endpoint       | Limit        | Window                  |
| -------------- | ------------ | ----------------------- |
| `POST /submit` | 20 requests  | per minute per API key  |
| `POST /submit` | 500 requests | per day per API key     |
| `POST /appeal` | 3 requests   | per hour per author_id  |
| `GET /log`     | 60 requests  | per minute (admin only) |

### Reasoning

**20 req/min on `/submit`:** The Groq LLM call is the bottleneck — at 20 req/min we stay well within free-tier Groq rate limits while allowing a platform to process a moderate burst of simultaneous submissions. A platform submitting content in real time (users posting) is unlikely to exceed this; a bulk ingestion pipeline should use a queue rather than direct API calls.

**500 req/day on `/submit`:** Prevents a single API key from consuming the entire Groq quota. Legitimate platforms processing high volumes should contact us for a higher tier.

**3 req/hour on `/appeal`:** Appeals are a human action. Three per hour is generous for any real creator — the limit exists to prevent automated flooding of the review queue.

### Implementation

Rate limiting is implemented using `Flask-Limiter` with an in-memory store for development. Limits are applied via decorators on each route. Exceeded limits return HTTP 429 with a `Retry-After` header.

---

## Known Limitations

**Casual, plainly-punctuated human writing is a predictable blind spot.** The stylometric signal scores "AI-likeness" partly on exotic punctuation density (em-dashes, ellipses, semicolons) and on type-token ratio sitting in a particular middle band. A piece of writing that is genuinely human but happens to use only periods, exclamation points, and commas — which describes a lot of ordinary casual writing, including texts, social captions, and informal blog posts — will score artificially AI-like on this signal regardless of how clearly human its voice is. I confirmed this directly: a casual, personal piece of writing scored 0.70+ on the stylometric signal alone, purely because it had no exotic punctuation, even though the content (personal references, specific names, conversational asides) was unmistakably human to a reader. The system partially recovers because the LLM signal is weighted higher and can catch what the stylometric signal misses, but if the LLM signal is _also_ uncertain on the same text, the result lands in `uncertain` rather than `human` — which is a defensible outcome (the system is honest about its limits) but not necessarily the outcome a confident human creator would want to see on their own writing.

**Short-form content (under ~100 words) gets unreliable stylometric scores by design, but the system doesn't yet enforce a hard floor.** Sentence length variance and burstiness need enough sentences to be statistically meaningful; on a 3-sentence submission, a single unusually short or long sentence can swing the score dramatically. `planning.md` anticipated this and proposed a mandatory `uncertain` label below 100 words, but that floor is not yet implemented in code — it's a known gap between the design and the current implementation (see Spec Reflection below).

---

## Spec Reflection

**Where the spec helped guide implementation:** the requirement that "a 0.51 confidence should produce a meaningfully different transparency label than a 0.95" forced a real design decision rather than letting me default to a naive weighted average. Without that requirement, the disagreement penalty in the fusion layer probably wouldn't exist — a simple average would have passed a basic "does it return a float" check, but it would have hidden exactly the kind of signal conflict the system is supposed to surface. Having that bar in the spec meant I had to verify, with actual test cases, that those two specific values produced different label variants, not just different decimal numbers.

**Where the implementation diverged from the spec:** `planning.md` specifies a mandatory `uncertain` label for any submission under 100 words, regardless of signal scores, to guard against the stylometric engine's unreliability on short text. I have not implemented this floor in `submit.py` — the word count is available but currently unused for this purpose. I deprioritized it because it touches the route logic rather than either signal in isolation, and I wanted to verify both signals and the fusion math independently first. It's the most concrete gap between the design document and the current code, and the right next step if I were continuing this project.

---

## AI Usage

This project was built collaboratively with Claude (Anthropic), used as a pair-programming and design-review tool throughout. Two specific instances worth detailing:

**1. Fusion algorithm — directed, then corrected for spec drift.** I asked Claude to implement the confidence fusion function exactly according to the weights, penalty factor, and thresholds I had specified in `planning.md`. The first version it produced was reasonable-looking but I explicitly asked it to verify the output against the spec's threshold table rather than accept it at face value — this caught nothing wrong in that instance, but it's a habit I kept for every milestone afterward, because earlier in the project an endpoint path (`/analyze` vs `/submit`) and a Groq model name had drifted from what we'd agreed without either of us noticing until a later review pass.

**2. Stylometric scoring on a real submission — used to diagnose, not just confirm.** When I wanted to check whether a specific casual piece of writing would score as human-leaning, I asked Claude to run it through the actual signal functions rather than reasoning about it abstractly. The first version of that text scored 0.696 on the stylometric signal — clearly AI-leaning, which surprised me given how obviously human the writing read. Rather than accepting that as "the model is just imperfect," I asked Claude to show me exactly which sub-signal was driving the score (`exotic_punctuation_density` came back 1.0, meaning zero em-dashes/ellipses/semicolons detected). I then asked it to revise the text by adding natural punctuation variation and re-run the diagnostic, rather than just asserting it would work — we iterated twice, checking actual scores each time, before landing on a version that pushed the fused score into the human-leaning range. The final text and its score breakdown are documented in this README as a real example of the stylometric blind spot, not a hypothetical one.

In both cases, the discipline was the same: ask the AI to produce something against an explicit, checkable spec, then verify the output by running it rather than reading it and assuming correctness.
