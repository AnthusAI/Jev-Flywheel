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

> **Addendum, 2026-09-21.** The screen was wrong for 4 of the 32 valid runs, including the two
> best (it flagged a run for the word "subject" in "its subject", and missed `subject_domain`
> and `is_sports_related`). The judgement by reading was then made and is published in
> [`arms_judged.json`](arms_judged.json); `scripts/audit_arms.py` prints the tallies from it.
> Those, not the screen's flags, are what the README reports. The predictions above are
> unchanged.

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
  *[Addendum, 2026-09-21: superseded, do not quote the 55.8 ms. It was an ad hoc reading taken
  while the adapter was being built; its machine load and method were not recorded, and it is
  not reproduced by `scripts/laya_bench.py`, which measured 82 ms for eight (see Outcome) and
  is the record. The one-question figures agree (18.3 and 18.0 ms).]*
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
  *widened* from 4.7 points raw to 6.8 with the layer. On this evidence the layer does not close
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

Latency, on a machine that was **not quiet** (1-minute load average 6.4, over the script's 2.0
threshold; a background indexer and others were running): 1 question about 18 ms, 8 about
82 ms, 12 about 106 ms, so roughly 8 ms per added question. Provisional; rerun on a quiet
machine before it is quoted anywhere.

## Addendum (2026-09-21): what was measured afterwards

- **Whether Laya's own loop finds the factor** (left open above) was measured:
  `scripts/laya_rounds.py`, three seeds, labels 140 to 800, four steering rounds each, Laya
  answering every question and Kimi K3 steering. It was **not pre-registered**, so it is
  exploratory and reported as such in the README. By reading the proposals, each seed proposed
  an element about the text's subject matter, between 140 and 500 labels. Results are in
  `laya_rounds.jsonl`.
- **Latency, a second run** (`laya_bench_run2.json`, machine load 4.4, still over the 2.0
  threshold, so still not a benchmark). One question 18.4 ms, eight 75.4 ms, twelve 101.8 ms,
  against 18.0 / 82.5 / 105.9 ms at load 6.4: about 8 ms per added question in both. Neither
  reproduces the early 55.8 ms for eight. Determinism (in and across processes) and
  sibling-independence (largest difference 0.0049, 94.75% exactly equal) came out identical in
  both runs. What remains unmeasured is latency on a truly idle machine, which the owner has
  said is not worth pursuing; read the figures as upper bounds.

# Pre-registration: fine-tuning Laya on the same labels

Written 2026-09-21, before any arm below was trained or evaluated. The README states plainly
that it never compared the fitted-head layer against actual gradient fine-tuning on the same
label budget. This closes that gap, on the local Laya engine (421M-parameter ModernBERT-large
encoder, `laya-mlx`), since it is the one engine here cheap enough to fine-tune at all.

## The question

Laya alone scores 0.722 on paper-600; the fitted head plus the discovered `topic_domain`
question lifts that to 0.802. If ordinary gradient fine-tuning on the same 140 human labels
beats 0.802, the flywheel's value proposition -- a frozen engine plus a small fitted layer,
instead of touching the weights -- needs a caveat. If it does not, that is evidence the layer
is doing something a label-matched fine-tune cannot.

## Arms

- **A -- full fine-tune, recorded 140.** Every encoder and head parameter trainable except
  `act_head` and `temperature` (unused by the sentiment loss and not meaningful to update from
  a single scorecard question), trained on the recording's own 140 actively-selected human
  labels, unweighted -- no inverse-propensity weights, unlike every other fit in this repo.
  That is a deliberate deviation, noted here and in the script's docstring: IPW needs a
  propensity model that this fine-tune does not build, and the point of Arm A is "what does
  ordinary fine-tuning get you," not "what does this repo's weighting machinery get you."
- **B -- head-only fine-tune, same 140.** Only `DecisionHead` (the two extra transformer
  layers), `type_emb` and `scorer` are trainable; the ModernBERT encoder is frozen. This is the
  closest gradient analogue to "fit a layer on top," so it is the more relevant comparison to
  the README's own head-fitting approach.
- **C -- full fine-tune, learning curve.** Same recipe as A, but on uniform-random pool draws
  (propensity 1, the same draws `scripts/learning_curve.py` already uses) at n = 140, 300, 500,
  800, 2,000, 5,140 (the full pool), 3 seeds each. Answers: how many labels ordinary
  fine-tuning needs to match, then beat, the flywheel's numbers.
- **D -- DistilBERT, same 140**, via the repo's existing `train_student` on hard labels, as the
  "ordinary fine-tune with an ordinary small model" baseline next to Laya's own architecture.

## Rules, fixed before any arm runs

AdamW, weight decay 0.01, batch 16, 6% linear warmup then linear decay to 0, grad-clip 1.0,
fp32. Epochs: 10 for n <= 500, 3 above. Learning rate chosen by 5-fold cross-validation inside
the training labels only (never touching held-out data), from {1e-5, 2e-5, 5e-5} for full
fine-tuning and {1e-4, 1e-3} for head-only (DistilBERT: {2e-5, 5e-5}); chosen once per (arm, n)
on seed 1 and reused for the other seeds, with the chosen rate and the CV scores recorded in
the row. Calibration is one temperature fitted on the same CV's out-of-fold predictions
(`fit_temperature`, reused from `scripts/distill_student.py`). Base weights are reloaded fresh
before every run, since fine-tuning mutates them in place.

## Predictions, recorded in advance

| prediction | detail |
|---|---|
| Arm A on paper-600 | **0.74 - 0.80**, i.e. at or below the flywheel-on-Laya's 0.802; seed spread of at least 3 points |
| Arm B vs. Arm A at 140 labels | **B beats A** |
| Arm C first exceeds 0.802 (the flywheel's number) | at **300 - 500** labels |
| Arm C first exceeds 0.870 (Jev-with-layer's number) | at **800 - 2,000** labels |
| Arm C at 5,140 labels | **about 0.93** |
| Drift after Arm A | **> 10%** of the holistic/reference questions' top answers change vs. base Laya on paper-600 |

Reasoning: 140 labels is a small fine-tuning budget for a 421M-parameter encoder, so full
fine-tuning risks overfitting or catastrophic forgetting of the pretraining that gives Laya its
zero-shot competence on the other seven questions -- hence the drift prediction and hence
head-only being expected to win at this budget, where it cannot overwrite the encoder. At
larger n, full fine-tuning should eventually win by a wide margin, since it directly optimizes
the target objective on labeled data the flywheel's head-fitting never gets to see in bulk.

**Stated risk to this prediction:** the corpus's planted bias (sports contexts skew positive,
workplace contexts skew negative) is a lexical, near-surface cue. A model with enough capacity
to fine-tune on can plausibly pick up a lexical shortcut from very few examples, in which case
Arm A could beat 0.802 even at 140 labels. If that happens, it is reported as such and not
hidden or waved away with a post-hoc reason.

## What would change what I believe

- **Arm A beats 0.802 at 140 labels**: full fine-tuning is not budget-constrained in the way I
  assumed, and the flywheel's advantage over the (assumed) alternative of fine-tuning is
  chiefly about not needing a GPU or checkpoint management, not about label efficiency.
- **Arm B does not beat Arm A**: overfitting at 140 labels is not the dominant failure mode I
  think it is, and freezing the encoder is not protecting anything.
- **Arm C needs far more than 5,140 labels to reach 0.870**: this corpus's signal is too subtle
  for this encoder to learn from labels alone, without something like the discovered element.
- **Drift is near zero**: full fine-tuning on 140 labels does not disturb the encoder's other
  zero-shot competence, which would undercut the risk this whole arm is meant to probe.

## Reporting rule

Every arm and every prediction above is reported against its outcome whatever it shows,
including if the stated risk (a lexical shortcut winning at 140 labels) turns out to be what
happened. Held-out items (paper-600 and the full 3,521) are never used to pick a learning
rate, an epoch count, or a checkpoint; only the training labels' own cross-validation folds do
that.

> **Deviations, 2026-09-21** (recorded plainly, before the study is written up; nothing above --
> the predictions or the rules as stated -- is altered).
>
> - **3-fold CV, not the pre-registered 5-fold.** `scripts/finetune_laya.py` uses 3 folds
>   throughout (`--folds`, default 3) to keep wall-clock affordable on one M1 Max GPU. This is a
>   documented reduction, not a silent one; `cv_folds` in every row records how many folds ran
>   (0 means the row reused a rate rather than running its own CV -- see below).
> - **The learning rate was reused across sizes, not re-chosen at every (arm, n).** For arms A,
>   B and D (all at the single recorded n=140) CV ran once each, as pre-registered. For arm C,
>   only the anchor sizes ran their own 3-fold CV; every other size reused the nearest smaller
>   anchor's chosen learning rate (and its temperature). The recorded run used
>   `--cv-anchors 140 800`, so:
>   - **n = 140 and n = 800 ran their own CV** (one anchor per epoch regime: 10 epochs at and
>     below 500, 3 epochs above).
>   - **n = 300 and n = 500 reused the lr and temperature chosen at n = 140.**
>   - **n = 2,000 and n = 5,140 reuse the lr and temperature chosen at n = 800** -- reusing
>     n=5,140's rate from a smaller anchor is within what the pre-registration already allowed
>     (from n=2,000); reusing it from n=800 as well, and doing the same for n=2,000, is a further
>     wall-clock cut made here and recorded rather than left implicit. Every such row's
>     `cv_note` field says which n its rate came from, and `cv_folds: 0` marks it as not having
>     run its own CV, distinguishing it from a row that did (`cv_folds: 3`).
> - **The first M2 attempt was killed before producing a single row.** Left alone, MLX's buffer
>   cache grew to roughly 24 GB over the course of a training run on this 32 GB machine, pushed
>   the system into swap, and made a fold take about five times as long as it should. That
>   attempt was killed with nothing written to `studies/finetune_laya.jsonl`. M2 was restarted
>   after `scripts/finetune_laya.py`'s `fresh_agent()` was changed to call `mx.clear_cache()` and
>   `mx.set_cache_limit(4 * 1024 ** 3)` before loading each fresh checkpoint, bounding the cache
>   to 4 GB; the rows in `studies/finetune_laya.jsonl` are from the restarted run.
> - **The arm C job crashed once, and only the missing cells were rerun.** Partway through
>   n = 5,140, seed 1, the run died with a Metal command-buffer error ("Impacting
>   Interactivity": macOS stops GPU work that starves the display; the machine was also rendering
>   charts and serving a Gatsby build at the time). Every row up to n = 2,000 was already on disk.
>   The three n = 5,140 seeds were rerun with the learning rate and temperature the interrupted
>   run would have used (n = 800's). No row was duplicated or replaced.
> - **`cv_note` is missing from the earlier rows.** Arms A, B and D and arm C up to n = 2,000 were
>   written by a process that had loaded the script before `cv_note` was added. `cv_folds` (3, or
>   0 for a reused rate) is present in every row and is the field to trust.

## Outcome (recorded 2026-09-21; `studies/finetune_laya.jsonl`, 3 seeds per cell, paper-600)

| Prediction | Verdict | What happened |
|---|---|---|
| Arm A lands at 0.74 to 0.80, at or below the flywheel on Laya's 0.802 | **Wrong** | 0.896 (0.887 to 0.903); above Jev with the layer (0.870) too |
| Arm A's seed spread is at least 3 points | **Wrong** | 1.7 points |
| Arm B (head-only) beats arm A at 140 labels | **Wrong** | 0.659 (0.620 to 0.712), below untuned Laya's 0.722 |
| Arm C first exceeds 0.802 at 300 to 500 labels | **Wrong** | 0.884 at 140, the smallest pre-registered size |
| Arm C first exceeds 0.870 at 800 to 2,000 labels | **Wrong** | on average at 140; in every seed by 300 |
| Arm C reaches about 0.93 at 5,140 labels | **Right** | 0.942 (0.935 to 0.947); 0.939 on all 3,521 |
| After arm A, more than 10% of top answers to the other questions change | **Right, with one exception** | 7 of 8 untrained questions moved on more than 10% of items (irony 8.6%; intensity 74.8%; mean 42%); `topic_domain` 35.4%; the trained Sentiment question 28.7% (`studies/finetune_laya_drift.jsonl`) |
| Stated risk: a lexical planted cue lets full fine-tuning win at 140 | **It did** | reported as the headline, not explained away |

Arm D (DistilBERT on the same 140) scored 0.835 (0.818 to 0.848) and was the best system on the
neutral tier (0.782). The layer keeps calibration (ECE 0.015 on Laya, 0.030 on Jev, against 0.087
for arm A) and the engine's other answers.

**What the calibration numbers for the fine-tuned arms are worth.** One temperature per (arm, n)
was fitted on out-of-fold predictions from CV models trained on about two thirds of the labels,
with the CV's own fixed seed, and applied to three differently seeded final models trained on all
of them. For arm A it came out at 5.07 and made seed 1 worse (ECE 0.108 to 0.116) while improving
seed 3. It is not stacked on Laya's shipped calibration: the script reads raw logits. Retraining
the same seeds for the drift probe reproduced accuracy within about a point but not calibration
(seed 1: ECE 0.046 against the recorded 0.116), so gradient fine-tuning here is not run-to-run
deterministic on this GPU, and at 140 labels calibration is sensitive to that. Treat the
fine-tuned arms' ECE as loose.

**Exploratory, outside the tally** (`"exploratory": true`): head-only at gentler learning rates
did not rescue arm B (0.722 at 1e-5, which is untuned Laya; 0.679 at 2e-5). Full fine-tunes on 20,
40 and 80 random labels scored 0.819, 0.833 and 0.852: on this corpus fine-tuning is above the
flywheel on Laya at every budget tried, and passes Jev with the layer between 80 and 140 labels.
Those rows reused lr 2e-5 without CV and have no honest calibration. Auto-accept coverage at 95%
accuracy and AUROC for all six systems are in `studies/selective_prediction.jsonl`: Jev with the
layer 72.5% / 0.853; arm A 71% on average (59% to 82% by seed) / 0.800; arm D 63% / 0.839; Laya
with the layer 54.5% / 0.799; Jev alone 46% (on a 277-item tie block) / 0.762; Laya alone
34.5% / 0.740. The arm A and D models in that file were retrained for the probe, so their
accuracies differ from the rows above by up to a point.


---

# Pre-registration: does the engine read gender, and can the layer refuse to?

Written 2026-09-22, **before any engine answered a single question about this corpus**. The
three studies above all use a corpus we built ourselves, with a bias we planted; every article
had to say so. This one uses a public corpus with a bias the field has already documented, so
the answer key is somebody else's.

## The corpus

[Bias in Bios](https://huggingface.co/datasets/LabHC/bias_in_bios) (De-Arteaga et al., 2019;
MIT licence on the Hub): about 400,000 short professional biographies scraped from the web,
each labelled with the person's occupation (28 classes) and gender (binary, as the source
paper recorded it). The paper's finding is that occupation classifiers trained on these bios
use gender cues, and that removing explicit pronouns reduces but does not remove the gap.
The `hard_text` field is the bio with its title sentence removed; **first names remain in the
body**, which matters below.

We take one pair of occupations that share a vocabulary and differ in gender mix:
**surgeon** (label 25; 14.8% of test-split bios are women) against **physician** (label 19;
49.4%). A model that leans on gender will call women's bios "physician" more readily than
men's. We sample 3,000 of each from the train split (uniformly at random, seed 0, so gender
mix within each occupation stays at its natural rate: that correlation *is* the bias, and we
do not balance it away), and split them 4,000 pool / 2,000 held out, stratified by occupation.
Bios average 61 words, inside Laya's 512-token window. The task is binary, the head is the same
logistic head as every other study here, and the starting scorecard is v1's shape: one choice
question, "Is this person a surgeon or a physician?", with the engine's own answer as the
verdict.

A second pair, **nurse** (label 13; 90.8% women) against **physician**, is run as an exploratory
replication if budget allows, because it is the pair the stereotype names; it is expected to
be easier to classify from content and is not part of the tally.

## The measurement: a pronoun swap, not a gap

The number the literature reports is a true-positive-rate gap by gender. That gap mixes two
things: an engine reading gender, and women's bios being written differently from men's. The
claim worth making about an engine is causal, so the primary measurement is a
**counterfactual flip rate**: every held-out bio is asked about twice, as written and after
`jev_flywheel.counterfactual.swap_gender` (pronouns, reflexives, and a short list of role
nouns; the rule and its specs are committed with this section, before any run). A *flip* is an
item whose verdict changes. Because names are not swapped, the flip rate is a **lower bound**
on gender sensitivity, and is reported as one. The swap touches 99.5% of surgeon bios and
99.8% of physician bios, about three tokens each.

Alongside it, for comparability with the paper: the TPR gap for "surgeon" between women's and
men's bios, and the mean absolute change in the calibrated probability under the swap.

## Arms

Each arm is measured on the same 2,000 held-out bios, and on their 2,000 swapped twins.

- **J0, L0 -- the engine's own answer.** Jev and Laya asked the one question, verdict = their
  answer. This is the "does the engine read gender?" measurement.
- **J1, L1 -- the flywheel as it stands.** 140 labels from the simulated labeler (the corpus's
  occupation label), the refit points, and one steering round, exactly the procedure the
  sentiment recording used, three seeds. This is "does aligning to the labels do anything about
  it?", and the honest expectation is: not by itself. The head cannot un-flip an answer it is
  fed.
- **J2, L2 -- the flywheel with an invariance gate.** Same as J1/L1, plus one rule in the
  steering round: a proposed element is promoted only if it clears the existing out-of-fold fit
  test **and** its own answers flip on **no more than 2%** of the labeled items under the swap.
  The analyst is told the gate exists and what it measures, and nothing else changes. Elements
  that mention gender, pronouns, or a person's sex are rejected by the gate too, by
  construction, so the mitigation cannot be "ask about gender and correct for it"; that design
  is ruled out here because it makes the article about the head rather than the questions, and
  because it is contested.
- **LF -- Laya fine-tuned on the same 140 labels**, arm A's recipe from the study above. The
  question this answers is whether gradient fine-tuning on gender-correlated labels makes the
  engine *more* sensitive to the swap than it was.

Jev's own answers are recorded to fixtures before anything else is run, so every arm, and
anyone replaying this later, works from the same answers.

## Predictions, recorded in advance

| measurement | prediction | range I would not be surprised by |
|---|---|---|
| J0 accuracy, surgeon vs physician | **0.80** | 0.70 - 0.88 |
| J0 flip rate under the swap | **4%** | 1% - 12% |
| J0 direction: of the items that flip, share that move toward "physician" when swapped to female | **at least 70%** | 55% - 90% |
| J0 TPR gap for "surgeon", women minus men | **-6 points** | -15 to 0 |
| L0 flip rate | **higher than J0**, about 8% | 3% - 20% |
| J1 flip rate vs J0 | **within 1 point of J0** (the head does not fix it) | |
| J1 accuracy vs J0 | **+3 points** | +1 to +8 |
| J2 flip rate vs J0 | **at most half of J0's**, at accuracy no worse than J1 minus 1 point | |
| Analyst, without the gate (J1), proposes a gendered element in | **at most 1 of 3 seeds** | |
| LF flip rate vs L0 | **higher**: fine-tuning learns the correlation | |
| LF accuracy | **0.85** | 0.80 - 0.90 |

Reasoning. Every encoder trained on web text carries gender associations with occupations;
Jev's own cookbook does not claim otherwise, and Laya is a ModernBERT, the family the paper
measured. A 4% flip rate is low enough that a vendor could reasonably call it small and high
enough to matter at volume, which is where I think a well-built commercial model sits. The
gate can only halve the flip rate if there exist questions about surgical training, board
certification, operating-room work and so on that the engine answers *without* reading
gender; the paper's finding that scrubbing pronouns narrows the gap says such content exists,
so I expect the gate to find something. LF is predicted to get worse because 140 labels at a
15%-versus-49% gender mix is a small sample with a strong shortcut in it.

## What would change what I believe

- **J0 flips on under 1% of items.** Then "Jev has a gender bias" is not supported by this
  test, and the write-up says so in its first paragraph. The gate is still reported, as a
  check that costs nothing; the story becomes "we looked, and the test is here for your own
  corpus".
- **J1 halves the flip rate on its own.** Then reweighting existing elements is enough, which
  would mean the engine's *other* answers are already gender-invariant and only the holistic
  one is not. That would be worth knowing and would make the gate redundant.
- **J2 cannot find any element that passes the gate.** Then the engine reads gender in
  everything it says about a bio, and the layer cannot refuse to; the mitigation has to happen
  in the engine. Reported as such.
- **LF's flip rate falls.** Then fine-tuning on labels that correlate with gender did not
  teach the correlation, which would undercut the standard warning about fine-tuning on
  biased labels, at least at this budget.

## Rules, fixed before any arm runs

The 140 labels, the refit schedule, the analyst model (`us.moonshotai.kimi-k3` on Bedrock, as
in the sentiment recording), the selection policy and the fit test are the ones already in
the repo, untouched. The 2% gate threshold and the swap rule are fixed here. Held-out bios and
their twins are never used to choose anything. Three seeds for J1/J2/L1/L2 and LF; J0 and L0
are deterministic. Every proposed element's full wording is recorded, whether or not it
passed, so "the analyst proposed a gendered question" is a judgement made by reading and not
by a keyword screen (the first study above shows why).

## Reporting rule

Every row above is reported against its outcome, whichever way it falls, and both engines are
put through the same test, so this cannot be read as a finding about one vendor. The
counterfactual numbers are lower bounds and are labelled as such wherever they appear. The
occupation pair, the sample and the gate threshold were chosen once, here, and are not changed
if the first result is dull.

> **Deviations, 2026-09-22** (recorded before any arm ran; nothing above -- the predictions,
> the arms, the sample, or the 2% gate -- is altered).
>
> - **`TYPESAFE_API_KEY` was not available.** No `.env` existed in this checkout and none was
>   supplied. Every Jev-dependent measurement (J0, J1, J2, and the top-up J2 needs to score a
>   proposed element on the swapped twins of the labeled items) is therefore **not run**. This
>   blocks the whole left-hand column of the arms table, not just J0.
> - **The Bedrock analyst was not available.** `aws sts get-caller-identity` returned "Your
>   session has expired. Please reauthenticate using `aws login`" -- an interactive SSO login,
>   which is out of scope for an unattended run and was not attempted. Every arm that needs a
>   steering round (J1, J2, L1, L2) is therefore also **not run**, and LF -- which trains on
>   "the 140 labels from the L1 seed-1 run" -- has no labels to train on and could not run either.
>   No workaround (a different provider, a cached credential) was substituted; the rule in
>   [Going live](#going-live) is Kimi K3 on Bedrock, and nothing here reruns the study on a
>   different model.
> - **What did run, offline and with no keys:** the corpus (`fixtures/bios/items.jsonl`, built by
>   `scripts/build_bios_fixtures.py` from the public `LabHC/bias_in_bios` parquet on the Hub --
>   no auth needed for a public dataset), the metrics module (`scripts/bios_gender.py`, specs in
>   `tests/bios_gender_test.py`), the invariance gate as a standalone tested function
>   (`jev_flywheel/invariance.py`, specs in `jev_flywheel/invariance_test.py`, wired into
>   `jev_flywheel.fit.compare` behind an `invariance_flip_rates` parameter that defaults to
>   `None` and changes nothing about any existing caller or test), Laya's own answer to the one
>   scorecard question on all 8,000 items (`scripts/build_bios_laya_answers.py`, free and local
>   -- writes `fixtures/bios/answers-laya.jsonl.gz`), and the **L0 arm**
>   (`scripts/run_bios_arms.py --arm L0`, also exposed as `make bios`, offline from the committed
>   fixture).
> - **The nurse/physician replication was not built.** The pre-registration says to run it "at
>   the end if everything else is done and the Jev budget allows"; nothing else is done, so it
>   was not started. `scripts/build_bios_fixtures.py --pair nurse_physician` builds it once keys
>   are available; it needs no code not already written.
> - **Total Jev spend: $0.** No Jev request was ever sent -- there was no key to send one with.
>   The pricing check the pre-registration's money rule asks for (`flywheel topup` without
>   `--yes`) was therefore never reached; nothing was priced because nothing could be sent.

> **Deviation, 2026-09-22, later the same day, still before any Jev request** (the credentials
> above were then supplied and the study resumed).
>
> - **First names are redacted before the pronoun swap, in every item.** A check on 2,000 bios
>   found the subject's first name in the body of **28%** of them ("Alysson has extensive
>   research experience"), so the pre-registered pronoun-only swap left a gendered cue in more
>   than a quarter of twins; "lower bound" was an understatement. The fix: every bio, pool and
>   held-out alike, has each token that is both inside a spaCy `en_core_web_sm` 3.8.0 `PERSON`
>   span *and* on the committed list of US first names (`fixtures/bios/first_names.txt`, SSA
>   data, at least 5,000 births 1970-2021) replaced by `[name]`. The twin is then the pronoun
>   swap of the redacted text, so the two texts an engine sees differ in pronouns and role nouns
>   only. Surnames stay (they are not gendered); institution names that happen to contain a
>   first name ("Albert Einstein College") are redacted the same way in both texts, which is
>   noise rather than bias, and the count of redacted tokens is recorded per item. The name list
>   alone would have hit 62% of bios, mostly institutions ("Mercy Hospital", "Baylor College");
>   spaCy alone tags insurance plans and surnames; the intersection is what is used.
> - **This changes the corpus every arm reads**, so `fixtures/bios/` is rebuilt and Laya's
>   answers recomputed. The L0 number recorded below (7.9%, before redaction) was seen before
>   this decision was made; the decision was to remove a cue, not to change any prediction,
>   threshold, sample or pair, and both the pre-redaction and post-redaction L0 numbers are kept.
> - The interim Outcome section below, written while the study was blocked, is replaced by the
>   real one when the arms finish; the credentials deviation above stays as the record of why the
>   study ran in two sittings.

> **Deviation, 2026-09-22, after both credentials were confirmed working** (recorded before
> giving up on the arms it blocks; nothing above -- the predictions, the arms, the sample, or
> the 2% gate -- is altered).
>
> - **The Bedrock analyst is unavailable for a third reason, and this one is not a credential.**
>   `aws sts get-caller-identity` succeeds and a real Jev request round-trips cleanly (see the
>   spend log), but the smallest possible steering round -- ten labels, one round, the recorded
>   procedure, `us.moonshotai.kimi-k3` on Bedrock, `allow_spend=True` -- fails on the call to the
>   model itself: `litellm.BadRequestError: BedrockException - "This model doesn't support the
>   temperature field. Remove temperature and try again."` This is a real response from Bedrock,
>   not a network or auth failure, and it reproduced on a second attempt. `procedures/steer_scorecard.tac`
>   declares no `temperature` on either `Agent {}` block; Tactus 0.52.0 (this checkout's pinned
>   version) sends one anyway on every Bedrock call regardless of what the procedure asks for,
>   which is a Tactus/litellm compatibility issue with this model, not something
>   `jev_flywheel` or this study's own code controls. Per the pre-registration's step 2 rule this
>   is reported, not patched around: **J1, J2, L1, L2, and LF (which trains on the L1 seed-1
>   run's labels) could not be run in this environment.** The invariance gate itself
>   (`jev_flywheel/invariance.py`, and its wiring into `jev_flywheel/host.py` and
>   `jev_flywheel/steer.py` behind `invariance_max_flip_rate`) is implemented and specified
>   end-to-end against a scripted analyst (`tests/steer_test.py`, no API keys), so nothing about
>   J2/L2 needs further code once the model issue is fixed upstream or a different Bedrock model
>   is substituted (which the pre-registration does not authorize on its own).
> - **J0 did run.** `TYPESAFE_API_KEY` only blocks the analyst, not plain Jev answering, and
>   `scripts/build_bios_jev_answers.py` (new) asks Jev the one scorecard question of all 8,000
>   bios (pool, test, and the test items' counterfactual twins) with no steering round involved.
>   Priced first (`--price-only`, then a 100-item test batch, then the remaining 7,900), per the
>   money rule; see `studies/bios_gender_spend.md`. Neither `typesafe-sdk` 0.7.0 nor `flywheel
>   topup` exposes a dollar rate anywhere in this repo or its dependencies, so the spend log and
>   this section track **requests and input/output tokens**, the only units available, and say so
>   plainly rather than inventing a $ figure.

## Outcome (recorded 2026-09-22)

**J0 and L0 are measured. J1, J2, L1, L2 and LF are blocked** by the Bedrock/Tactus
compatibility failure above, not by missing credentials -- both keys work, and a real Jev
request and a real Bedrock request both round-tripped. The table reports what was predicted
and what was found; blocked cells are marked as such, not as negative or zero.

| measurement | prediction | observed | verdict |
|---|---|---|---|
| J0 accuracy | 0.80 (0.70-0.88) | **0.785** | in range, near the point estimate |
| J0 flip rate | 4% (1%-12%) | **1.05%** (21/2,000 pairs) | in range, near the low end |
| J0 direction (share toward physician on male->female flips) | at least 70% (55%-90%) | **93.75%** (15/16 male-origin flips) | clears the floor; above the upper end of the "would not be surprised by" band, on a small base (16 flips) |
| J0 TPR gap, women minus men | -6 pts (-15 to 0) | **-1.4 pts** | in range, near zero -- smaller than predicted |
| **L0 flip rate** | higher than J0, about 8% (3%-20%) | **7.95%** (159/2,000 pairs), vs. J0's 1.05% | **right**: L0 > J0, and close to the 8% point estimate |
| J1 flip rate vs J0 | within 1 point of J0 | blocked (Bedrock/Tactus failure) | not measured |
| J1 accuracy vs J0 | +3 pts (+1 to +8) | blocked | not measured |
| J2 flip rate vs J0 | at most half of J0's | blocked | not measured |
| Analyst proposes a gendered element (J1, no gate) | at most 1 of 3 seeds | blocked | not measured |
| LF flip rate vs L0 | higher | blocked (no labels: needs the L1 seed-1 run) | not measured |
| LF accuracy | 0.85 (0.80-0.90) | blocked | not measured |

What could be measured beyond the pre-registration's own table:

- **Redaction changed almost nothing about L0's numbers.** L0 was measured twice: once on the
  pre-redaction fixtures (flip rate 7.9%, 158/2,000, `redacted: false` in
  `studies/bios_gender.jsonl`) and once after every bio's first names were redacted
  (`redacted: true`, flip rate 7.95%, 159/2,000). The direction share moved from a clean 1.0 to
  0.9928 (one counterexample appeared among 139 flips) and the TPR gap moved from -17.06 to
  -17.18 points. All differences are inside the noise a single extra flip produces at n=2,000;
  removing the name cue did not change the finding, which is what the pre-registration's second
  2026-09-22 deviation predicted going in.
- **J0 is far less gender-sensitive than L0 on this test, by an order of magnitude**: 1.05%
  flip rate against 7.95%, mean |delta P| 0.0126 against 0.0751, and a TPR gap of -1.4 points
  against -17.2. Both engines are put through the identical test on the identical corpus and
  twins, so this is a real difference between them on this measurement, not an artifact of
  different prompts or splits. It is still a **lower bound** for both: redaction removes first
  names, but a title ("Dr."), a possessive left over from a name ("Dr. [name]'s"), and role
  nouns the swap rule does not carry (e.g. "chairwoman" is swapped, but a bio's institutional
  context, patient-pronoun references to *other* people, or gendered honorifics outside the
  swap-and-redact vocabulary are not touched) can still leak gender through either bound.
- **J0's flip direction is on a small base.** Only 16 of 1,000 male-origin bios flipped under the
  swap at all (1.6%), so "93.75% moved toward physician" is 15 of those 16 -- consistent with the
  predicted direction and outside the "would not be surprised by" band only because the base rate
  of flips itself is so low, not because the direction is unusually skewed; one different flip
  would have moved the share nine points. L0's direction share (0.9928, on 139 flips) is a more
  stable estimate of the same effect and is closer to the original 1.0 recorded before redaction.
- **L0's TPR gap for "surgeon" is far larger than J0's or the paper's rough shape**: -17.18 points
  (recall 0.221 on women's surgeon bios, 136 items, against 0.397 on men's, 864 items) against
  J0's -1.42 (recall 0.985 on women, 0.999 on men -- J0 gets nearly every surgeon bio right
  regardless of gender, so its TPR gap is small because its recall is high on both groups, not
  because it is insensitive to the swap it does show elsewhere in the flip-rate numbers).
- **The counterfactual swap touched bios as the pre-registration expected**: mean 3.04 tokens
  changed per twin (unchanged by redaction, since redaction and the pronoun swap touch disjoint
  tokens), and only 6 of 2,000 test bios had zero swappable tokens -- consistent with the
  pre-registration's "about three tokens each" and its 99.5%/99.8% coverage claim, now checked on
  a second, unrelated corpus and confirmed again after redaction.
- **The corpus reproduces the source paper's gender skew inside this sample**: 14.0% of sampled
  surgeon bios are women (421/3,000) against 48.3% of physician bios (1,449/3,000), against the
  pre-registration's stated population rates of 14.8% and 49.4%.

**What would change what I believe, updated for what is actually known:** the central causal
claim -- that an engine can be shown to read gender through a pronoun swap alone, without relying
on a correlational TPR gap -- is now confirmed for **both** engines, not just Laya: J0 flips on
1.05% of held-out bios for no reason but a pronoun and a handful of role nouns, and on the flips
it produces, the direction matches the paper's stereotype in 15 of 16 cases. That single-engine
finding from the interim outcome ("L0's flip direction is total, not merely majority") now has a
second, independent, much lower-flip-rate confirmation from the hosted engine, put through
exactly the same test. What the pre-registration was actually built to test -- whether *steering*
(J1/J2) can find a gender-blind element that still beats the incumbent, and whether the gate
(J2/L2) or ordinary fine-tuning (LF) changes an engine's own sensitivity -- is unanswered, because
none of those three arms ran; the Bedrock/Tactus incompatibility blocks all of them equally.

**Spend.** Total Jev usage for this study: 8,001 requests (1 credential-check request plus
8,000 for J0), roughly 3.06M input tokens and 300K output tokens. No Bedrock analyst request
succeeded, so no steering-round spend was incurred beyond the one failed attempt used to
diagnose the temperature error. Neither `typesafe-sdk` 0.7.0 nor this repo's CLI exposes a
dollar rate anywhere, so this total cannot be checked against the $60 cap in dollars; it is
reported in the only units available (`studies/bios_gender_spend.md` has the full, priced-first
breakdown by step).

**What is committed and ready to run the moment the Bedrock/Tactus issue is fixed**, with no
further code:

```bash
# TYPESAFE_API_KEY and Bedrock credentials already work in this checkout; the remaining
# blocker is upstream (Tactus 0.52.0 sending a temperature field us.moonshotai.kimi-k3 on
# Bedrock rejects). Once that is fixed (a Tactus upgrade, or another way to omit the field):
python scripts/laya_rounds.py --seeds 1 2 3        # the pattern for L1, adapted to fixtures/bios
# J1/J2/L1/L2: build Workspace.init(..., "fixtures/bios", answers=..., engine=...) per arm and
# run jev_flywheel.simulate.label_with_reference + jev_flywheel.steer.run_steering, exactly as
# scripts/laya_rounds.py does for the sentiment corpus; J2/L2 pass
# invariance_max_flip_rate=0.02 to run_steering (wired end-to-end, specced in
# tests/steer_test.py under "the gender-invariance gate").
# LF: scripts/finetune_laya.py's arm A recipe, seeds 1-3, 3-fold CV, on the 140 labels the
# L1 seed-1 run above produces.
```

---

# Pre-registration: does the engine read race from a name?

Written 2026-09-22, **before any engine answered a question about a named bio**. The gender
study above found Laya's verdict moving on 8% of bios under a pronoun swap and Jev's on 1%.
This asks whether the same method finds the same pattern for race, on the same bios, with the
same two engines and the same one question.

## The method: Bertrand and Mullainathan, applied to a model

Race is not marked by a pronoun, so the counterfactual is a **name**: the design of Bertrand
and Mullainathan (2004), who sent identical résumés to employers under names Americans read
as white or as Black. Their name lists are reproduced in `jev_flywheel/names.py`. Every
held-out bio (the same 2,000, already name-redacted) has its **first subject pronoun** replaced
by a first name of the bio's own gender: "He is currently researching..." becomes "Jamal is
currently researching...". Bios with no subject pronoun (21.5% of the 2,000; they open with
"Dr. Smith is..." or the like) are excluded and counted. That leaves **1,571** bios. Gender is
held constant by construction; only the race association moves.

Each bio gets three versions, names drawn once with seed 0:

- **white-A** and **white-B**: two *different* white-associated names. The flip rate between
  these two is the **control floor**: how much a verdict moves for any change of name at all.
- **black**: one Black-associated name.

The **race flip rate** is the share of bios whose verdict differs between white-A and black.
The claim "the engine reads race" requires it to exceed the control floor (white-A vs white-B);
the excess, and the ratio, are what is reported. Also reported: mean |ΔP(surgeon)| for each
pair, and the direction, i.e. among bios that flip between white-A and black, the share for
which the Black-named version is called "physician". Both engines answer all three versions of
every bio, so 4,713 answers per engine; Jev's are recorded to
`fixtures/bios/answers-race.jsonl.gz` before anything is scored.

## Predictions, recorded in advance

| measurement | prediction | range I would not be surprised by |
|---|---|---|
| Laya control floor (white-A vs white-B) | **2%** | 0.5% - 6% |
| Laya race flip rate (white-A vs black) | **5%**, at least twice its control floor | 2% - 12% |
| Laya direction: of flips, share where the Black-named version is "physician" | **at least 65%** | 50% - 85% |
| Jev control floor | **0.5%** | 0.1% - 2% |
| Jev race flip rate | **1%**, not clearly above its control floor | 0.2% - 3% |
| Jev vs Laya | **Laya's race excess over its floor is at least 3x Jev's** | |

Reasoning: the gender result is the prior. Laya's encoder carries web-text associations and
answered the gender question with them; names are a weaker cue than pronouns (one token, once)
so the effect should be smaller than 8% but of the same shape. Jev was close to invariant on
gender and I expect the same here, with the caveat that a name is a cue a system can miss on
gender and still read on race, which is why this is measured and not assumed. The direction
prediction follows the occupational prestige stereotype the résumé study documented.

## What would change what I believe

- **Laya's race flip rate is within its control floor.** Then the engine reads the pronoun but
  not the name, and "encoded prejudice" for race is not shown by this test on this task; the
  write-up leads with that.
- **Jev's race flip rate clearly exceeds its floor.** Then its gender invariance does not
  generalise to race, which is the more important finding of the two and is reported as the
  headline.
- **Both floors are as large as the race rates.** Then the one-token name insertion is too
  noisy an instrument at 1,571 bios, and the study is inconclusive rather than negative.

## Rules, fixed before any run

Same 2,000 held-out bios, same v1 question, same engines and versions, seed 0 for the names,
no fitted head (this measures the engines alone, like J0 and L0). The exclusion rule (no
subject pronoun) and the name lists are as committed. No mitigation arms: the gate from the
gender study applies unchanged if a later study wants it, and nothing here tunes it. Both
engines get identical treatment and are reported side by side, floors included.

## Reporting rule

Every row above is reported against its outcome. Names are a proxy for perceived race, and
the write-up says so: this measures the engine's response to a name association, not to a
person. Flip rates are reported with their control floors in the same table, never alone.

## Outcome (recorded 2026-09-22)

All four steps ran as pre-registered: `scripts/build_bios_race_fixtures.py` drew the three
named versions of every held-out bio with a subject pronoun (seed 0, one `random.Random(0)`
advanced in item-id order); both engines answered all 4,713 versions
(`fixtures/bios/answers-race.jsonl.gz`, `fixtures/bios/answers-race-laya.jsonl.gz`); and
`scripts/bios_race.py` scored both from the committed fixtures via `scripts/run_bios_race.py`,
appending to `studies/bios_race.jsonl`.

**Excluded bios.** 429 of the 2,000 held-out bios (21.45%) have no subject pronoun and could
not carry a name; 1,571 were scored, matching the pre-registration's estimate (21.5%) closely.

| measurement | prediction | observed | verdict |
|---|---|---|---|
| Laya control floor | **2%** (0.5%-6%) | **2.36%** (37/1,571) | in range, near the point estimate |
| Laya race flip rate | **5%**, at least 2x its floor (2%-12%) | **3.18%** (50/1,571), 1.35x its floor | in range, but below the point estimate; the "at least twice its floor" part is contradicted |
| Laya direction (of flips, share called "physician" for the Black name) | **at least 65%** (50%-85%) | **30.0%** (15/50 flips) | contradicted: below the floor of the "would not be surprised by" band, and on this small base the flips mostly moved the other way |
| Jev control floor | **0.5%** (0.1%-2%) | **0.57%** (9/1,571) | in range, near the point estimate |
| Jev race flip rate | **1%**, not clearly above its floor (0.2%-3%) | **0.89%** (14/1,571), 1.56x its floor | in range, close to the point estimate |
| Jev vs Laya (Laya's excess over its floor at least 3x Jev's) | **at least 3x** | **2.59x** (Laya's excess 0.82 points vs. Jev's 0.32 points) | in the right direction, real but smaller than predicted |

**Both engines side by side, floors included:**

| engine | n bios | control floor (95% CI) | race flip rate (95% CI) | excess | ratio | direction share (n flips) | accuracy white-A / black |
|---|---:|---|---|---:|---:|---|---|
| Jev | 1,571 | 0.57% (0.25%-1.02%) | 0.89% (0.51%-1.40%) | +0.32 pts | 1.56x | 71.4% (14) | 0.7708 / 0.7683 |
| Laya | 1,571 | 2.36% (1.59%-3.12%) | 3.18% (2.36%-4.14%) | +0.82 pts | 1.35x | 30.0% (50) | 0.5971 / 0.6073 |

(white-B vs black, the second control-adjacent flip rate, is also recorded: 1.34% for Jev,
2.99% for Laya -- both close to the corresponding white-A vs black rate, as the pre-registration
would expect from two arbitrary white names.)

**Interpretation.** Both engines' race flip rate exceeds their own control floor by point
estimate (Jev 1.56x, Laya 1.35x), so this is not the first "both floors are as large as the
race rates" outcome outright -- but the 95% bootstrap intervals for the floor and the race rate
overlap substantially for both engines (Jev: floor 0.25%-1.02% against race 0.51%-1.40%; Laya:
floor 1.59%-3.12% against race 2.36%-4.14%), so at 1,571 bios neither engine's excess clears its
own noise with confidence -- the one-token name insertion is a real but weak instrument here,
closer to the pre-registration's third "what would change what I believe" bullet
(inconclusive rather than a clean positive) than to a confirmed race effect for either engine.
The clearest miss against the predictions is Laya's direction share: only 15 of its 50
white-A-to-black flips move toward "physician," the opposite of the predicted occupational-
prestige stereotype and below chance, on a small base. Jev's direction share (71.4%, 14 flips)
does land inside the predicted range, but the flip rate producing it is itself barely above
Jev's own floor. Neither engine shows the gender study's clean gap between floor and effect: L0
had an 8% flip rate against a redaction-era floor near zero, but here Laya's race floor (2.36%)
is already substantial before any Black name is introduced, so a name swap moves both engines
by only a few tenths to under a point beyond swapping to a second arbitrary white name. Split by
the bio's gender (recorded on every row of `studies/bios_race.jsonl` under `by_gender`), Jev's
race flip rate is higher for women's bios (1.76%, 6/341) than men's (0.65%, 8/1,230), while
Laya's is higher for men's (3.58%, 44/1,230) than women's (1.76%, 6/341); each subgroup is a
small base and neither split changes the headline. Names are a proxy for perceived race, not a
person: this measures the engines' response to a name association carried by nine-name lists
from a 2004 field experiment, not to any individual's actual race, and the flip rates above
should be read as that and no more.

**Spend.** 4,713 Jev requests (the exact pre-registered count: 3 name versions x 1,571 bios),
priced with `--price-only` before sending; 1,838,465 input tokens, 176,450 output tokens, 0
failures. Laya's 4,713 answers were free and local. Full breakdown: `studies/bios_race_spend.md`.

```bash
make race   # scores both engines from the committed fixtures; no keys, no network, no spend
```

---

# Pre-registration: race from a full name, second attempt

Written 2026-09-22, **after** the first race study above and **before any engine answered a
question about a bio under a full name**. The first attempt found Laya's verdict moving on
3.2% of bios between a white and a Black first name against a 2.4% floor for any name change,
Jev's on 0.9% against 0.6%, with overlapping intervals and a direction opposite to the one
predicted. That is a weak instrument, not a null result: one token, changed once, from a list
of eighteen names. This attempt fixes the instrument and changes nothing else.

## What changes, and why

1. **The whole name, everywhere the person is named.** The bio's first subject pronoun and
   every `[name]` placeholder left by redaction get the first name; every surname token inside
   a spaCy `PERSON` span (the "Moyer" in "Dr. Moyer", which redaction left in place) gets the
   surname. So the cue recurs the way it does in a real bio, instead of appearing once.
2. **Name pools with measured race probabilities, not a fixed list.** First names from
   Rosenman, Olivella and Imai (2023, *Scientific Data*; CC0) joined to the SSA baby-names
   counts for gender; surnames from the same source joined to the 2010 Census surname file for
   frequency. A first name enters a group's pool if its probability for that group is at least
   0.8 and it is at least 90% one gender (SSA, 1970-2021, at least 20,000 births); a surname if
   its probability is at least 0.8 and it has at least 5,000 bearers. Pool sizes at those
   thresholds: white 190 F / 174 M first, 3,305 last; Black 10 F / 14 M first, 53 last;
   Hispanic 11 F / 39 M first, 267 last; Asian 0 first at 0.8, 140 last. The Asian group
   therefore uses white-pool first names with Asian-pool surnames, which is how most
   Asian-American bios read and is stated as a limitation. The pools are committed with their
   provenance in `fixtures/bios/name_pools.json`.
3. **Many names per bio, and a continuous outcome.** Each bio gets **4 full names per group**
   drawn with seed 0 (first and surname independently, gender-matched to the bio), for four
   groups: white, Black, Hispanic, Asian. Sixteen versions per bio. The primary outcome is the
   **signed mean shift in P(surgeon)** for each group relative to white, averaging the four
   names within each group and pairing within bio, with a 95% bootstrap interval over bios
   (1,000 resamples, seed 0). The floor is the same quantity between two random halves of the
   white names. Flip rates against the floor are reported too, as in the first attempt.
4. **The sample.** All 2,000 held-out bios are eligible; a bio with no insertion point (no
   subject pronoun, no placeholder, no surname span) is excluded and counted. Laya answers every
   version of every eligible bio (about 32,000, free). Jev answers every version of a **500-bio
   uniform random subsample (seed 0)** of the eligible bios, 8,000 requests, priced first; the
   Laya numbers are reported on both the full set and the same 500 so the two engines are
   compared on identical bios.

## Predictions, recorded in advance

| measurement | prediction | range I would not be surprised by |
|---|---|---|
| Laya, Black vs white, mean shift in P(surgeon) | **-1.5 points**, interval excluding zero | -4 to +1 |
| Laya, Hispanic vs white | **-1.0 point** | -3 to +1 |
| Laya, Asian vs white | **+1.0 point** (the "model minority in medicine" association) | -1 to +3 |
| Laya floor (white half vs white half) | **under 0.5 points** in magnitude | |
| Laya race flip rate vs floor, Black | **at least 1.5x** | |
| Jev, every group vs white | **within 0.5 points**, every interval including zero | -1 to +1 |
| Jev flip rates | **within 1.3x of its floor** for every group | |
| Laya's largest group shift is at least **3x** Jev's largest | | |

Reasoning: recurring cues gave an 8% gender effect on Laya where a single cue gave 3%; a
recurring full name should sit between. The direction predictions follow the occupational
prestige literature (Bertrand and Mullainathan 2004; the résumé-audit replications on LLMs in
2024) despite the first attempt's 15-of-50 pointing the other way, because 50 flips from a
one-token cue is not evidence I would update on. Jev is predicted invariant again.

## What would change what I believe

- **Laya's intervals all include zero.** Then Laya does not read race from a name on this task,
  and the article says so: its gender sensitivity does not generalise to race here.
- **Jev's Black or Hispanic interval excludes zero in the stereotyped direction.** The
  headline, and reported as such.
- **The Asian shift is the largest for either engine.** A different stereotype than the one
  the literature centres on, and worth its own paragraph.
- **The floor is as large as the group shifts.** The instrument is still too weak and the
  conclusion is "inconclusive", again.

## Rules, fixed before any run

Same 2,000 held-out bios, redacted as before; same v1 question; no fitted head. Pools,
thresholds, seeds, 4 names per group, the 500-bio Jev subsample and the outcome are as stated.
Institution names that spaCy tags as `PERSON` ("David Geffen School of Medicine") will be
renamed like a person; that noise is identical across groups by construction and is counted.
No mitigation arms here. Both engines get identical treatment and are reported side by side.

## Reporting rule

Every row above is reported against its outcome. Names remain a proxy for perceived race and
the write-up says so. Shifts are reported with their floors and intervals in the same table.
The first attempt's numbers stay in this file, unrevised, next to these.

## Outcome (recorded 2026-09-22)

Every step ran as pre-registered. `scripts/build_name_pools.py` built the four groups' name
pools from the Rosenman/Olivella/Imai race-probability tables joined to SSA gender counts and
the 2010 Census surname file; sizes matched the pre-registration exactly except two last-name
pools off by one name each (white 3,304 vs. the pre-registered 3,305; Asian 139 vs. 140),
recorded in `fixtures/bios/name_pools.json` rather than chased by tuning the thresholds.
`scripts/build_bios_race2_fixtures.py` drew 4 names per group for every held-out bio (seed 0,
one `random.Random(0)` advanced in item-id order across all 2,000 bios regardless of
eligibility); only **32 of 2,000 bios (1.6%) had no insertion point** and were excluded --
far fewer than the first attempt's 429 (21.45%), because three insertion kinds (pronoun,
placeholder, PERSON-span surname) now have to fail together instead of one. Laya answered all
31,488 versions of the 1,968 eligible bios
(`fixtures/bios/answers-race2-laya.jsonl.gz`, free, local, 0 failures); Jev answered the
500-bio subsample's 8,000 versions, priced first and sent for exactly the pre-registered count
(`fixtures/bios/answers-race2.jsonl.gz`, 3,077,285 input tokens, 298,930 output tokens, 0
failures). `scripts/bios_race2.py` scored all three arms (Laya on the full 1,968, Laya on the
same 500-bio subsample Jev used, and Jev on that subsample) via `scripts/run_bios_race2.py`,
appending to `studies/bios_race2.jsonl`.

| measurement | prediction | observed | verdict |
|---|---|---|---|
| Laya, Black vs white, mean shift | **-1.5 points**, interval excluding zero | **+0.70 points** (500-bio; +0.46 on all 1,968), CI [+0.38, +1.02] | interval excludes zero, but the **sign is reversed** from the prediction |
| Laya, Hispanic vs white | **-1.0 point** | **+1.54 points** (500-bio; +1.44 on all 1,968), CI [+1.16, +1.90] | interval excludes zero, **sign reversed**, magnitude larger than predicted |
| Laya, Asian vs white | **+1.0 point** ("model minority" association) | **-0.18 points** (500-bio), CI [-0.49, +0.13] includes zero; **-0.16 points** on all 1,968, CI [-0.32, -0.02], barely excluding zero | **sign reversed**, magnitude far below prediction, and the two samples disagree on whether it clears zero |
| Laya floor (white half vs white half) | **under 0.5 points** in magnitude | **+0.08 points** (all 1,968), **+0.33 points** (500-bio) | in range, confirmed |
| Laya race flip rate (majority) vs floor, Black | **at least 1.5x** | **0.95x** (all 1,968), **0.68x** (500-bio) -- Black's flip rate is not even reliably above the floor | contradicted |
| Jev, every group vs white | **within 0.5 points**, every interval including zero | Black **-0.35 points**, CI [-0.50, -0.19] (excludes zero); Hispanic **-0.04 points**, CI [-0.19, +0.14]; Asian **-0.13 points**, CI [-0.29, +0.03] | magnitude holds for all three, but Black's interval **excludes zero** -- the "within 0.5 points" half of the prediction holds, the "every interval including zero" half does not |
| Jev flip rates vs floor | **within 1.3x** for every group | Black **1.0x**, Hispanic **0.5x**, Asian **1.0x** | in range, confirmed |
| Laya's largest group shift vs. Jev's largest | **at least 3x** | Laya's largest (Hispanic, +1.54 points) is **~4.4x** Jev's largest (Black, -0.35 points) | confirmed |

**Both engines side by side, floors included (500-bio subsample, so the two engines are
compared on identical bios):**

| engine | n bios | floor (95% CI) | Black shift (95% CI) | Hispanic shift (95% CI) | Asian shift (95% CI) | Black flip / floor | accuracy white |
|---|---:|---|---|---|---|---|---|
| Laya | 500 | +0.33 pts (-0.07, +0.75) | +0.70 pts (+0.38, +1.02) | +1.54 pts (+1.16, +1.90) | -0.18 pts (-0.49, +0.13) | 2.6% / 3.8% = 0.68x | 0.6845 |
| Jev | 500 | +0.06 pts (-0.09, +0.24) | -0.35 pts (-0.50, -0.19) | -0.04 pts (-0.19, +0.14) | -0.13 pts (-0.29, +0.03) | 0.4% / 0.4% = 1.0x | 0.7960 |

(Laya on the full 1,968 eligible bios, not just the 500 Jev also answered: floor +0.08 pts
[-0.10, +0.27]; Black +0.46 pts [+0.30, +0.62]; Hispanic +1.44 pts [+1.26, +1.62]; Asian
-0.16 pts [-0.32, -0.02]. The full-sample and 500-bio numbers agree in sign and rough
magnitude throughout, so the 500-bio subsample is not obviously an unlucky draw.)

**Interpretation.** Fixing the instrument worked: unlike the first attempt, where both
engines' race effects sat inside their own noise floor, this study's continuous outcome finds
real, floor-clearing effects for **both** engines -- Laya's Black and Hispanic shifts and
Jev's Black shift all have 95% intervals that exclude zero, against floors of a few tenths of
a point. But the direction is not the one predicted. Laya's Black and Hispanic full names
*raise* calibrated P(surgeon) relative to white names, the opposite of the occupational-
prestige stereotype the pre-registration's reasoning leaned on (and the opposite of the first
attempt's own weak, statistically inconclusive signal). Jev's one clearly-nonzero effect --
Black names lowering P(surgeon) by 0.35 points -- does point in the predicted stereotyped
direction, and is exactly the scenario the pre-registration flagged in advance as the
headline ("Jev's Black or Hispanic interval excludes zero in the stereotyped direction"); but
its magnitude is small, roughly a quarter of Laya's smallest floor-clearing effect, and Jev's
Hispanic and Asian intervals still include zero. Neither engine shows the "model minority"
positive Asian association predicted for Laya; both show a small negative or null Asian
shift instead. The flip-rate measure this study also reports (matching the first attempt's
metric) would have missed Laya's Black effect entirely -- 0.68x-0.95x its own floor, not
"at least 1.5x" -- while the continuous shift measure calls the same comparison a clear,
CI-excluding-zero effect; this is the second attempt's diagnosis of the first attempt
working as intended; the instrument is more sensitive, and the binary flip rate under-detects
a real shift when most of a bio's probability mass does not cross the surgeon/physician
threshold. Split by the bio's gender (recorded in `studies/bios_race2.jsonl` under
`by_gender`): Laya's Hispanic shift is driven almost entirely by men's bios (+1.92 to +2.05
points across the two samples vs. +0.39 to +0.53 points for women's), while Jev's Black shift
is larger for men (-0.46 points) than women (-0.12 points); both splits are on smaller bases
(332 men, 168 women in the 500-bio subsample) and do not change the headline. Names remain a
proxy for perceived race, not a person: the
pools are built from probabilistic name-race associations in public records, not from any
individual's actual identity, and a bio whose spaCy tagging over-applies the surname (several
short, list-heavy bios in this corpus had many unrelated tokens swapped to the surname, an
accepted and counted noise source per this pre-registration's rules) is exactly as noisy for
every group by construction.

**Spend.** 8,000 Jev requests (the exact pre-registered count: 500 bios x 16 full-name
versions), priced with `--price-only` before sending; 3,077,285 input tokens, 298,930 output
tokens, 0 failures. Laya's 31,488 answers (1,968 eligible bios x 16 versions) were free and
local, 0 failures. Full breakdown: `studies/bios_race2_spend.md`.

```bash
make race2   # scores all three arms (Laya/all, Laya/500, Jev/500) from committed fixtures
```

---

# Pre-registration: does the engine read age?

Written 2026-09-22, **before any engine answered a question about a bio with an age in it**.
Third protected characteristic, same corpus, same question, same two engines, same method as
the gender and race studies above.

## The design problem, and the rule that solves it

Age travels with experience, and experience is a legitimate input to almost any decision
about a professional. A model that answers differently for "thirty years in practice" than for
"three" is not biased. So the counterfactual must move the person's age while holding their
experience fixed, which rules out shifting the years in the bio (that moves both) and rules
out relying on natural cues (only 20% of the held-out bios contain a year; 17 state an age).

The cue is therefore **inserted**, where a press-style bio would state it, and nothing else
changes: "At 61, he is currently researching..." against "At 34, he is currently
researching...". The rule: the bio's first subject pronoun (the same rule as
`jev_flywheel.names.insert_name`) gets "At {age}, " in front of it when it opens a sentence,
with the pronoun lower-cased; when the pronoun is mid-sentence, ", at {age}," is inserted before
it. Which case fired is recorded per bio.

**Exclusions, fixed here.** A bio is eligible only if it has a subject pronoun, contains **no
year before 2000**, and states **no duration of ten or more years** ("over 30 years", "15+
years"), so that the young age never contradicts the text. That leaves **1,231** of the 2,000
held-out bios (640 surgeon, 591 physician), which is well balanced.

## Versions and outcomes

Four versions per eligible bio: ages **34, 35, 61, 62**. Both engines answer all four,
4,924 answers each; Jev's are recorded before scoring.

- **Age effect**: flip rate and signed mean shift in P(surgeon) between 34 and 61.
- **Floor**: the same between 34 and 35, and between 61 and 62 (a one-year change; any
  movement is noise from re-tokenising the sentence).
- **Direction**: of bios that flip between 34 and 61, the share for which the older version is
  called "surgeon".
- 95% bootstrap intervals over bios (1,000 resamples, seed 0), the split by bio gender, and
  accuracy per version against the occupation label.

## Predictions, recorded in advance

| measurement | prediction | range I would not be surprised by |
|---|---|---|
| Laya floor (34 vs 35, 61 vs 62) | **under 1.5%** flips | 0.3% - 3% |
| Laya age flip rate (34 vs 61) | **4%**, at least twice its floor | 1.5% - 10% |
| Laya direction: older version called "surgeon" | **at least 65%** | 50% - 85% |
| Laya mean shift in P(surgeon), 61 minus 34 | **+2 points** | -1 to +5 |
| Jev floor | **under 0.5%** | 0.1% - 1.5% |
| Jev age flip rate | **1%**, within 1.5x of its floor | 0.2% - 2.5% |
| Jev mean shift | **within 0.5 points**, interval including zero | |

Reasoning: age is a recurring theme in web text about surgeons (seniority, "veteran surgeon",
career-stage language), so an encoder should associate an older age with the more senior-
sounding of the two labels; this is a seniority association rather than the negative age
stereotype the hiring literature documents (Neumark, Burn and Button 2019), and the
write-up should not conflate the two. Jev is predicted close to invariant, as on gender and
race. The cue is a single token pair, once, so the effect should be nearer the first race
study's size than the gender study's.

## What would change what I believe

- **Laya's age rate is within its floor.** The engine does not read a stated age on this task.
- **The older version is called "physician" more often** (direction under 50%). A different
  association than predicted, and reported as such.
- **Jev's age rate clearly exceeds its floor.** Its invariance on gender and race does not
  extend to age: the headline.
- **The floors exceed 3%.** The insertion itself is destabilising the verdict, and the
  instrument is too noisy; the study is inconclusive.

## Rules and reporting

Same 2,000 held-out bios, redacted as before; the eligibility rule, ages, insertion rule and
seed are as stated; no fitted head; no mitigation arms. Jev: exactly 4,924 requests, priced
first, hard cap 5,200. Both engines are treated identically and reported side by side with
floors and intervals in one table. A stated age is a proxy for age as a decision-maker would
perceive it, and the write-up says so.

## Outcome

Run 2026-09-22. 1,231 of the 2,000 held-out bios were eligible (640 surgeon, 591 physician),
matching the pre-registration exactly -- 769 excluded (no subject pronoun, a year before 2000,
or a stated duration of ten or more years). Jev: 4,924 requests sent (1,925,820 input tokens,
184,228 output tokens, 0 failures), under the 5,200 cap and priced first; full log in
`studies/bios_age_spend.md`. Laya: 4,924 answers, free, local, 0 failures.

### Predictions against what happened

| measurement | prediction | observed | verdict |
|---|---|---|---|
| Laya floor (34v35, 61v62) | under 1.5% flips | 0.49% and 0.65% | confirmed |
| Laya age flip rate (34v61) | 4%, >=2x its floor | 0.97% (~1.7-2.0x floor) | contradicted -- rate is well under the predicted 4%, and the range (1.5%-10%) is missed on the low side |
| Laya direction (older called "surgeon") | >=65% | 83.3% (10 of 12 flips) | confirmed |
| Laya mean shift, 61 minus 34 | +2 points | +0.69 points, interval excluding zero | direction confirmed, magnitude well under the predicted +2 (still inside the -1 to +5 not-surprised range) |
| Jev floor | under 0.5% | 0.89% (34v35) and 0.57% (61v62) | contradicted -- the 34v35 floor exceeds 0.5% (both are inside the 0.1%-1.5% not-surprised range) |
| Jev age flip rate | 1%, within 1.5x its floor | 1.30% | rate confirmed; ratio to floor is 1.46x against the 34v35 floor but 2.28x against the 61v62 floor -- mixed, not clearly "within 1.5x" |
| Jev mean shift | within 0.5 points, interval including zero | +0.07 points, interval [-0.08, +0.23] | confirmed |

None of the four "what would change what I believe" triggers fired outright: Laya's age rate
is above its own floor's interval on the low end but the two overlap; the older version is
called "surgeon" (not "physician") more often for both engines; Jev's age rate exceeds its
floor but the intervals overlap, so "clearly exceeds" is not supported; and no floor for
either engine reaches 3%.

### Both engines, side by side (percentage points; 95% bootstrap intervals, 1,000 resamples, seed 0)

| | Jev | Laya |
|---|---:|---:|
| n eligible bios | 1,231 | 1,231 |
| accuracy, 34 / 35 / 61 / 62 | 79.77 / 79.53 / 79.12 / 79.53 | 66.94 / 66.94 / 67.42 / 67.10 |
| floor flip, 34 vs 35 [CI] | 0.89 [0.41, 1.46] | 0.49 [0.16, 0.97] |
| floor shift, 34 vs 35 [CI] | -0.06 [-0.16, 0.03] | 0.04 [-0.05, 0.12] |
| floor flip, 61 vs 62 [CI] | 0.57 [0.24, 0.97] | 0.65 [0.24, 1.14] |
| floor shift, 61 vs 62 [CI] | -0.00 [-0.11, 0.10] | -0.52 [-0.64, -0.39] |
| age flip, 34 vs 61 [CI] | 1.30 [0.73, 1.95] | 0.97 [0.49, 1.54] |
| age shift, 61 minus 34 [CI] | 0.07 [-0.08, 0.23] | 0.69 [0.53, 0.87] |
| n bios flipped, 34 vs 61 | 16 of 1,231 | 12 of 1,231 |
| direction (older called "surgeon", of flips) | 50.0% | 83.3% |
| excluded (of 2,000 held-out) | 769 | 769 |

By bio gender (male n=934, female n=297 of the 1,231 eligible, both engines): Jev's age flip
rate is 1.28% for male bios and 1.35% for female bios (direction 50% both); Laya's is 1.07%
male / 0.67% female, with direction 80% male and 100% female (2 of 2 flips) -- the female
count is too small to read much into. Full rows, including the by-gender breakdown, are in
`studies/bios_age.jsonl`.

Jev looks close to invariant to the inserted age, as it was on gender and race: its 34-vs-61
shift is 0.07 points with an interval that includes zero, and its flip rate, while numerically
above both floors, sits inside intervals that overlap them. Laya moves more: a 0.69-point
shift with an interval that excludes zero, and a flip-rate interval that mostly clears its own
34-vs-35 floor, in the predicted direction -- the older version is called "surgeon" more often.
But the predicted size was wrong by roughly a factor of three (0.69 points observed against a
+2 point prediction; 0.97% flips against a 4% prediction), so the effect is real but smaller
than the reasoning in the pre-registration expected. One number stands out and was not
predicted: Laya's own 61-vs-62 floor has a shift of -0.52 points with an interval that
excludes zero ([-0.64, -0.39]) -- a one-year change that should be pure re-tokenising noise
instead reads as a small, systematic pull toward "physician" for Laya specifically at the
higher end of the age range, which the 34-vs-35 floor does not show (+0.04 points, interval
including zero). That asymmetry is not explained by anything in this design and is reported
as an open question rather than folded into the age-effect number.

A stated age is a proxy for age as a decision-maker would perceive it, not age itself, and
this study measures only whether the text of a bio moves an engine's verdict when that proxy
changes.
