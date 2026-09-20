# Jev Flywheel

> **TL;DR** Jev answers many typed questions about a text in one cheap request. Put a small
> model of your own on top of those answers and it can decide better, and say how sure it is
> more honestly, than Jev's own answer. Then let humans review its mistakes one at a time, let
> a language model read *why* it was wrong and propose better questions, and refit. This repo
> is the smallest working version of that loop, and you can replay a real run of it on your
> laptop in under a minute with no API keys.

![Four panels: held-out accuracy and calibration error by scorecard version, a reliability diagram, and agreement with the labeler over time](images/flywheel.png)

*Held-out results from one recorded run. Every version is scored on the same 600 items that no
labeler ever saw. The labeler in this recording is simulated: see [what this proves and what
it doesn't](#what-this-proves-and-what-it-doesnt).*

| Scorecard | Accuracy | Calibration error (ECE) | Brier |
|---|---|---|---|
| v1: Jev alone | 0.768 | 0.151 | 0.188 |
| v2: after a refit on 33 labels | 0.767 | 0.060 | 0.161 |
| v3: after one steering round, 140 labels | **0.853** | **0.036** | **0.102** |

Two different mechanisms did two different jobs. **A refit fixes calibration** and cannot
change accuracy: it only re-weighs the answers Jev already gave. **Steering fixes accuracy**,
because it changes which questions are asked.

This is the third article in a series: [Making Decisions Instead of Generating
Text](https://anth.us/blog/making-decisions-instead-of-generating-text/), then [Can You Trust
Jev's Confidence?](https://anth.us/blog/can-you-trust-jev-confidence/) (with its repo,
[Jev-Calibration](https://github.com/AnthusAI/Jev-Calibration)). This one is what you build
once you know Jev's confidence needs correcting and you have people who can tell you when it is
wrong.

## Try it

```bash
git clone https://github.com/AnthusAI/Jev-Flywheel && cd Jev-Flywheel
make install     # a virtualenv with everything, including Tactus
make demo        # replay the recorded run offline and redraw the figure
```

`make demo` needs no keys, no network, no model and no person. It rebuilds a workspace from the
committed fixtures, feeds back 140 recorded judgements, and re-runs each recorded refit and
steering round, including the language model's exact reply and the labeler's decision. Fitting
is deterministic, so it reproduces the same scorecards.

To label something yourself:

```bash
.venv/bin/flywheel init        # a workspace from the bundled 8,801-item corpus; offline
.venv/bin/flywheel label       # the console: agree or disagree, with an optional comment
.venv/bin/flywheel status      # what the system thinks is worth doing next
.venv/bin/flywheel evaluate    # held-out accuracy, and agreement with you
```

## The idea

Jev is a model from [TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
that takes some text and a set of typed questions (yes/no, choice, or a score on a rubric) and
returns a probability-backed answer to each, **in one request**. Its cost is dominated by the
text, not the questions. In this corpus eight questions averaged 501 input tokens per item
against about 335 for two.

That makes a decomposition affordable that would not be with ordinary LLM prompts, where every
extra question is another call. So instead of asking Jev the one question you care about and
trusting the answer, ask it that question *plus a handful of sub-questions*, and let a small
model of your own weigh them:

| Word | Meaning |
|---|---|
| **Score** | What you want decided: "Is this text positive or negative?" |
| **Element** | A sub-question Jev answers in the same request: "Is the text sarcastic?" Evidence, not a verdict. |
| **Feature** | One number the small model weights, derived from an element's answer. A yes/no gives one; a choice with K options gives K − 1. |
| **Decision** | The small model: features in, a value and a confidence out. Its weights are readable and live in the YAML. |
| **Scorecard** | One YAML file for every score. It mirrors Jev's architecture, where one request covers everything. |

```yaml
- name: Sentiment
  question_type: choice
  instructions: What is the overall sentiment of this text?
  criteria: {positive: null, negative: null}
  elements:
    - key: first_evaluative_polarity
      question_type: choice
      instructions: Setting aside any later reassurance or doubt, is the first evaluative statement ...
      criteria: [positive_first, negative_first, no_evaluation]
  decision:
    model: multinomial_logistic
    classes: [positive, negative]
    features: [self.holistic.clr.positive, first_evaluative_polarity.clr.positive_first, ...]
    parameters:
      weights: {positive: {intercept: 0.31, self.holistic.clr.positive: 1.84, ...}}
    calibration: {method: temperature, temperature: 1.31, ...}
```

The vocabulary (item, feedback item, element, feature, decision, the label sources, the label
normalization rules) is deliberately the same as [Plexus](https://github.com/AnthusAI/Plexus)'s,
so a scorecard and a feedback set made here move there as a port, not a rewrite.

## The flywheel

```
        ┌────────────────────────────────────────────────────────────┐
        ▼                                                            │
  pick the item a label       you agree or disagree,      refit the head (cheap, often)
  would teach the most  ───►  and say why, if you like ───►  and when the errors have a shape
  (active selection)                                        that better questions could fix,
                                                            a steering round (Tactus)
```

**Active selection.** Labels are the scarce resource, so the console does not show items in
file order. It scores every unlabeled item on how much a label would teach: the head is unsure,
the head and Jev's own answer disagree, the evidence pulls both ways, the item is far from
anything already labeled. Selection is stochastic on purpose, and it
records the probability each item was picked with, so the fit can undo the bias it introduces.

**Be skeptical of this part.** On the recorded run the tiers the labeler was shown
(12.1/19.3/47.1/21.4%) are indistinguishable from the pool's own composition
(11.1/17.5/48.6/22.7%) — at 140 labels, active selection did nothing measurable here. What is
demonstrated is the *plumbing*: propensities are recorded, so the fit can correct for whatever
the policy does. The policy earning its keep is unproven. An earlier version also penalized
items whose answers were all unsure, on the theory that they were irreducibly ambiguous; that
theory was wrong on this corpus, and the penalty is now recorded but not scored.

**Refit and steering are different operations, so they have different triggers.** A refit takes
milliseconds and only changes numbers, so it runs every few labels and is promoted only if it
beats the incumbent *out of fold*. A steering round costs an LLM call and possibly a Jev
top-up, so it waits until the residual looks structured: enough disagreements you explained,
and progress has plateaued, which says the bottleneck is the questions, not the sample size.
`flywheel status` lays out every condition and how far off it is.

**Steering is a Tactus procedure.** [Tactus](https://github.com/AnthusAI/Tactus) owns the loop
in [`procedures/steer_scorecard.tac`](procedures/steer_scorecard.tac): one analyst turn, a
cheap repair loop if the proposal is malformed, a price check before any Jev spend, one
evaluation, a human approval gate, and a checkpointed apply. The deterministic parts (reading
the feedback, fitting, pricing, committing) are Python behind a small host module. The language
model is asked one thing: read the disagreements and comments and propose edits.

What it proposed in the recorded run, from 140 disagreements, given only their text and the
element answers:

> The scorecard relies on a single holistic sentiment question, and the disagreements show it
> is systematically fooled by surface tone. In every hedged/contrastive text, the human labels
> by the **first** evaluative statement about the subject (criticism-first = negative even
> when a reassuring coda follows; praise-first = positive even when doubts follow), while the
> holistic answer follows the trailing clause. Separately, the human treats purely neutral,
> procedural wording with no evaluative content as negative.

It added two elements, `first_evaluative_polarity` and `contains_evaluative_language`, and
retired none. Held-out accuracy went from 0.767 to 0.853.

## Design decisions worth stealing

- **The agent proposes edits, code applies them.** The analyst returns a small JSON proposal
  (add, retire or reword elements). It has nowhere to put a weight, so a language model is never
  in the numeric path. A proposal is one batch by construction, which matters because any change
  to the question set costs a fresh Jev pass over the labeled items. A malformed proposal fails
  with a message the agent can act on before anything is fit or spent. (This also sidesteps a
  practical problem: Kimi K3 on Bedrock ignores Tactus's structured-output protocol and answers
  in prose, but returns clean JSON when asked.)
- **The answer cache is keyed per question, not per question set.** Adding an element then
  costs one request per labeled item carrying *only* the new question. Ten new elements cost
  the same number of requests as one, because requests are per item.
- **Features have a frozen contract.** Choice answers use a centered log-ratio, not per-option
  logits, because probabilities sum to one and per-option logits are collinear: regularization
  splits weight between them arbitrarily and two refits on the same data disagree. Probabilities
  are clipped at 0.01 because Jev's tails are not calibrated and answers flip about 1% of the
  time between identical runs.
- **Missing features are zero when serving and an error when training.** In log-odds space zero
  means "no evidence", so a degraded request degrades gracefully. But training on imputed rows
  biases a new element's weight toward zero, and then the optimizer concludes its own proposal
  was useless and retires it.
- **Calibrate on out-of-fold predictions only.** Isotonic regression memorizes what it is fit
  on. The calibrator accepts only a type that cross-validation constructs, so in-sample
  calibration cannot happen by accident. Method is gated on sample size: temperature scaling on
  a few dozen labels, isotonic only when there are thousands.
- **A capability ladder.** What the fit may do depends on the *effective* sample size (Kish's,
  after inverse-propensity weighting): hold, then a heavily regularized head, then a
  cross-validated one, then richer calibration. Feature count is budgeted against it, and asking
  for more is refused with a message saying what would suffice. The tiers are a frozen, named
  policy so a recorded fit stays interpretable.
- **The agent is never shown the held-out scoreboard.** It sees out-of-fold numbers only. An
  agent that saw the test split, even indirectly, would tune to it one round at a time.
- **The element inventory reports permutation importance, not coefficients.** An earlier
  experiment on this corpus found the largest coefficient belonged to an element that merely
  marked which tier an item was in. An agent reading raw coefficients would have concluded that
  was the key concept.

## What this proves and what it doesn't

**It proves the machinery, end to end, and the measurement.** The loop runs, the selection is
legible, the fit is honest about how much evidence it has, and each step is deterministic and
replayable. The two-part result (calibration from refits, accuracy from steering) is real and
reproducible from the recording.

**It does not prove that human comments surface hidden factors, because the labeler here is
not a human.** It answers with the corpus's own reference label and explains each disagreement
with a fixed template ("This is really negative; the wording is weak and it misleads."). The
analyst therefore diagnosed the pattern from the texts and the element answers, not from human
insight. The claim that matters, that a person's comments name factors nobody declared, needs
real labels. That is what `flywheel label` is for.

Other things to hold in mind:

- **The analyst is not deterministic.** Two live runs against the same 600 held-out items gave
  +2.4 and +8.5 points, because Kimi K3 proposed different elements each time. The recording
  commits one of them. Expect variance, and treat any single round as one draw.
- **600 items is roughly ±1.5 points** on a paired comparison.
- **The corpus is constructed, and its labels encode a factor that is not sentiment** (see
  above). It is a good test bed and a poor guide to how a messy real feedback set behaves.
- **One run rewrote the holistic question as well as adding an element,** so the gain cannot be
  attributed to the element alone. The analyst is now told to prefer elements, and the human is
  shown that rewording the holistic question makes every stored answer to it stale.
- **A refit on 33 labels can move calibration a lot and accuracy not at all**, by design. If
  yours moves accuracy, look at why before believing it.

## Going live

Everything above the fold is offline. Live mode needs two things:

```bash
cp .env.example .env    # TYPESAFE_API_KEY for Jev; the file is gitignored
```

- **Jev**, for `flywheel topup` and `flywheel steer --allow-spend`. Anything that would spend
  is priced first and asks before sending; `--allow-spend` is the explicit opt-in.
- **An LLM for the analyst**, on AWS Bedrock by default. The default model is **Kimi K3**,
  invoked through its inference profile (`us.moonshotai.kimi-k3`), because K3 is
  profile-only on Bedrock. It spends hidden reasoning tokens, so `max_tokens` is generous. If you
  use credentials from `aws login`, boto needs `botocore[crt]` (included in the `steer` extra).
  Tactus also supports OpenAI: `flywheel steer --provider openai --model gpt-4o`.

```bash
flywheel steer                  # one round; needs a labeled workspace
flywheel steer --allow-spend    # let it call Jev to evaluate a proposal that needs new answers
flywheel record my-run/         # export your session so someone else can replay it offline
```

The demo recording is regenerated with
[`scripts/make_demo_recording.py`](scripts/make_demo_recording.py), which costs a few cents.

## When you outgrow this

This is deliberately small, and it says so about what it leaves out: no database, no API, no
accounts, no dashboard, no job queue, no artifact storage, one score at a time, and a head that
is a logistic model with a handful of features. [Plexus](https://github.com/AnthusAI/Plexus) is
the industrial version of the same ideas, for when you need:

- **Scale and availability**: many scorecards and scores, evaluation runs, background workers,
  a store that survives a laptop.
- **Compliance**: multi-tenant accounts, audit trails, and controlled access to feedback.
- **Richer models**: on a related experiment, gradient boosting overtook the logistic head at
  roughly 500 labels. Anything past a handful of inline weights needs somewhere to keep them.
- **A whole feedback workflow**: reviewers, vetted labels, and sampling by confusion cell.

The [Anthus AI Solutions](https://anth.us) team builds and runs it. If this repo was useful and
you would like to take it further, get in touch.

## Layout

```
jev_flywheel/
  items.py        Item, FeedbackItem, label sources and Plexus's label normalization
  scorecard.py    the whole-scorecard YAML: load, validate, round-trip
  features.py     the frozen answer-to-feature contract
  head.py models.py   serving: standard library only, weights readable in the YAML
  jev.py answers.py   one request per item; a cache keyed per question
  fit.py calibrate.py ladder.py sampling.py evaluate.py   the honest fit
  selection.py steering.py   which item to ask, and when a steering round is worth it
  proposal.py host.py steer.py inventory.py   the analyst, its proposals, and the Tactus runner
  loop.py console.py cli.py workspace.py   the human-facing loop
  report.py charts.py recording.py   measurement, the figure, record and replay
procedures/steer_scorecard.tac   the steering loop, in Tactus
fixtures/         8,801 items, cached Jev answers, the recorded run
```

`make test` runs the specs (457, none needing a network or a key). The procedure's specs
are pytest-driven rather than Tactus BDD, because they need the Python host module registered,
which `tactus test` cannot do.

## License

MIT. The sentiment corpus is the public dataset from
[Jev-Calibration](https://github.com/AnthusAI/Jev-Calibration).
