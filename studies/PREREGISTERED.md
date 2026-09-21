# Pre-registration: can the loop find the sports/workplace factor?

Written **before** the runs, so the result can disappoint the prediction.

## The target

The corpus's authors planted a factor and published it: "sports contexts occur more often
among positive examples, while workplace contexts occur more often among negative examples."
An oracle element asking that question directly is worth about +9 points of held-out accuracy
over the refit baseline, the largest single lever measured in this project. The question is
whether the loop proposes it **unprompted**.

## Arms (4 models x 3 seeds each; one held-out sample per seed, shared across arms)

- **d0** — the loop as it stands: one analyst that sees the scorecard and the disagreements.
- **d1** — d0 plus a *blind pass*: a second agent shown only two groups of texts, "Group A"
  and "Group B", with no task name, no scorecard and no criteria, asked what separates them.
  Candidates from both passes are pooled and screened by the fit.
- **d2** — d1 plus a taxonomy of convention kinds in the analyst prompt (scope, exceptions,
  subject matter, register, thresholds). Subject matter is on that list, so d2 is a nudge and
  is reported as one.

## Predictions, recorded in advance

| arm | names the axis | mean paired gain |
|---|---|---|
| d0 | 1/12 | +4.5 pts |
| d1 | **6/12** | +6 pts |
| d2 | 8/12 | +6 pts |

Reasoning: frame-lock is the diagnosed obstacle and d1 removes the frame for the half of the
work that needs it gone, so d1 should be the large step and d2 a smaller one. If d2 is much
better than d1, the taxonomy did the work and must be disclosed as leading.

## What would falsify the diagnosis

- **d1 names it in <= 2/12** — frame-lock is not the obstacle. The cause would be the evidence
  or the sample size, and the fix is neither framing nor architecture.
- **d1 names it often but paired gain stays < +3 pts** — naming and value are decoupled; the
  "surfaced convention" is then a deliverable in its own right and not a route to accuracy.
- **d0 ~ d1 ~ d2** — the loop's ceiling is set by something none of these touch.

## Reporting rule

Every arm is reported whatever it shows, including this file's predictions against the
outcome. The detector is a keyword screen; each proposal's full wording is recorded so the
judgement is made by reading.

---

# Pre-registration: the same layer on Laya (a local 421M encoder)

Written 2026-09-21, **before any Laya answer was generated for this corpus**. The only Laya
numbers in hand when this was written are the ones its authors publish (near-chance zero-shot
on their typed-decisions benchmark, ECE 0.466 raw), none on sentiment.

## The question

Jev is a large hosted model. Laya is a 421M encoder that runs on this laptop, for free, in tens
of milliseconds. **Can the layer this repo builds -- a fitted head, calibration, and a loop that
discovers new elements -- close the gap between them?**

Reference points on the same 3,521 held-out items, from the Jev runs: Jev's own answer 0.760,
Jev + fitted head 0.843, Jev + head + the discovered subject-matter element about +10 points on
top of a refit.

## Predictions, recorded in advance

| measurement | prediction | range I would not be surprised by |
|---|---|---|
| raw Laya, holistic answer, accuracy | **0.66** | 0.55 - 0.75 |
| raw Laya, holistic ECE | **> 0.15** (over-confident) | 0.08 - 0.30 |
| Laya + fitted head (140 labels, same elements) | **0.72** | 0.66 - 0.80 |
| Laya + head + the subject-matter element | **0.80** | 0.72 - 0.86 |
| Jev-with-layer minus Laya-with-layer, at the end | **4 - 8 points** | 0 - 12 |
| the subject-matter factor helps Laya *more* than it helped Jev | **yes**, by 2+ points | |

Reasoning for the last two: subject matter (sports vs. workplace) is topic classification,
which an encoder does well, and the sentiment judgement Laya does poorly is precisely what the
head down-weights once a topic factor is available. So the layer should help the weaker engine
proportionally more, and still not fully close the gap.

## Measurements I have a prediction for that are about the engine, not the layer

- **Marginal cost of a question.** Already measured while building the adapter, so this is a
  record and not a prediction: 18.3 ms for one question and 55.8 ms for eight on this M1 Max
  (~3x for 8x the questions), against Jev where an extra question is a few input tokens.
- **Sibling-independence.** The tempting claim is that Laya answers each question as its own row,
  so an answer cannot depend on which other questions were asked, and the per-question answer
  cache is sound by construction. **I predict that is not quite true**: rows are padded to a common
  length inside a batch and FP16 arithmetic is not associative, so an answer will differ in the
  last digits depending on its siblings. Predicted maximum absolute probability difference between
  a question asked alone and the same question asked with seven others: **> 0 and < 0.01.**
  If it is exactly 0, the construction claim stands. If it exceeds 0.02, the cache needs a
  set-fingerprint key on this engine.
- **Run-to-run determinism.** Same process, same input: **identical.** (Seen once already.)
  Across separate processes: **identical.**
- **Corpus fits the window.** Every item fits with room to spare; the binding cost is not the
  512-token window but the per-question re-encoding.

## What would change what I believe

- **Laya-with-layer within 3 points of Jev-with-layer**: the layer, not the base model, carries
  most of the value on this task. That would be the headline, and it would need to be checked
  against a task where the base model has to do more of the work before generalising it.
- **Laya-with-layer more than 12 points behind**: the base model's floor matters and the layer
  cannot lift a weak encoder here. Also a headline.
- **The discovered element gives Laya less than +3 points**: the discovery does not transfer
  across engines, which would mean an element is partly a property of the engine that answers it.
- **Raw Laya above 0.75**: I under-rated it, and the gap the layer has to close is small to begin
  with.

## Reporting rule

Every row above is reported against its prediction whatever it shows. The Jev numbers are from
the existing runs and are not re-run. Laya answers are generated once, locally, and kept beside
Jev's, so any later comparison can be re-derived from the two fixtures.

## Outcome (recorded 2026-09-21; the paired replay and the engine benchmark)

Raw data: `studies/laya_paired.jsonl`, `studies/laya_bench.json`, `fixtures/answers-laya.jsonl.gz`.
The Jev rows reproduce the README's table exactly (0.768 / 0.763 / 0.765 / 0.870), which is the
check that the replay machinery and the Laya rows are comparable. The Laya numbers below are on
the same 600 held-out items; "full test" is all 3,521, affordable only because Laya is free.

| measurement | predicted | observed | verdict |
|---|---|---|---|
| raw Laya holistic accuracy | 0.66 (0.55-0.75) | **0.722** (600) / 0.716 (3,521) | in range, above my point estimate |
| raw Laya holistic ECE | > 0.15 | **0.107** (600) / 0.103 (3,521) | **wrong**: in my range, but not over-confident by the margin I called |
| Laya + fitted head | 0.72 (0.66-0.80) | 0.730 (600) / 0.725 (3,521) | right |
| Laya + head + subject-matter element | 0.80 (0.72-0.86) | **0.802** (600) / 0.806 (3,521) | right |
| Jev-with-layer minus Laya-with-layer | 4-8 pts | **6.8 pts** (0.870 vs 0.802) | right |
| the element helps Laya *more* than Jev | yes, by 2+ pts | Laya **+7.2**, Jev **+10.5** (the steering step) | **falsified** |
| sibling-independence: max abs diff | > 0 and < 0.01 | **0.0049**; 94.75% of 1,200 comparisons exactly equal | right |
| order of the questions | (not predicted) | **0.0 exactly** | new fact: order never matters |
| determinism, in and across processes | identical | identical | right |
| corpus fits the window | yes | longest item 48 tokens, 475 to spare | right |

What the two wrong calls say:

- **The layer helped the weaker engine less, not more.** My reasoning was that a topic factor
  substitutes for the sentiment judgement Laya does poorly. The gap between the engines
  *widened* from 4.6 points raw to 6.8 with the layer. On this evidence the layer does not close
  the gap; it lifts both engines and leaves the weaker one behind. The recorded proposal was
  written by an analyst reading **Jev's** disagreements, so this measures whether a factor
  *transfers*, not whether Laya's own loop would find it. That second measurement has not been made.
- **Raw Laya is better calibrated than I assumed** (ECE 0.10 against Jev's 0.15 on the same items),
  which is not what its authors' benchmark suggested. laya-mlx ships the calibration
  temperatures with the checkpoint, so this is the shipped model and not a refit.

Things I did not predict and would not have called:

- **Steering costs Laya its medium tier**: 0.991 before, 0.840 after, while weak (0.671 to 0.787)
  and neutral (0.483 to 0.703) improve. Jev shows the same shape more mildly (1.000 to 0.953).
  The promotion gate accepted this because overall accuracy rose. Per-tier cells here are small
  (a few dozen to a couple of hundred items in a 600 sample), so this is a lead to check on
  the full 3,521 and not yet a finding.
- **A refit made Laya's calibration worse before it made it better**: ECE 0.107 at v1, 0.153 at
  v2 (37 labels, promoted), 0.095 at v3. The Jev lineage never regressed on ECE. Selection
  gated on out-of-fold metrics, and the held-out scoreboard disagreed at n = 37.

Latency, on a machine that was **not quiet** (1-minute load average 6.8, over the script's 2.0
threshold; a background indexer and others were running): 1 question about 17 ms, 8 about
82 ms, 12 about 106 ms, so roughly 8 ms per added question. Provisional; rerun on a quiet
machine before it is quoted anywhere.
