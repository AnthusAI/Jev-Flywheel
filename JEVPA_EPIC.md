# JEVPA: Pareto search for Jev scorecards

## Goal

Test whether keeping several useful question sets alive, then combining their
questions, improves a Jev Flywheel scorecard more reliably than the current
one-proposal steering round. The analyst still proposes only scorecard edits.
Jev answers the questions; code fits all weights and calibration; a person
approves the scorecard that would be served.

This is a research epic. A clear negative result completes it too.

**Status, 2026-09-24:** the cached-answer search machinery and exploratory
sentiment pilot are implemented; see [`studies/JEVPA_PILOT.md`](studies/JEVPA_PILOT.md).
The pilot did not show a Pareto advantage on the primary Brier metric or a
distinctive merged question set. The registered live comparison below remains
future work, and the Biased-Decisions extension is not triggered by this pilot.
Its frozen protocol is [`studies/JEVPA_LIVE_PROTOCOL.md`](studies/JEVPA_LIVE_PROTOCOL.md).

## The question to answer

At the same analyst and engine budget, does a small Pareto pool of scorecards
produce better held-out decisions than choosing the best candidate after each
round? Is any gain due to preserving complementary questions, rather than
merely generating more questions?

## Experiment

### Arms

1. **Current loop:** one batch of edits per steering round, evaluated and
   promoted through the existing gate.
2. **Batch plus greedy:** draw several distinct question proposals in one
   priced batch, fit each candidate, and carry forward only the best overall.
3. **JEVPA:** use the same proposal and evaluation budget as arm 2, but retain a
   capped pool of candidates that do well on different predefined slices.
   Try combining complementary question sets, refitting the head from scratch.

Arm 2 isolates the value of Pareto retention and recombination from the value
of simply trying more questions. Report actual Jev requests, input and output
tokens, analyst calls, and questions asked per item for every arm. Stop each
arm at the same predeclared budget; a common request count alone is not enough
if one arm sends many more question tokens.

### Candidate rules

- A candidate is a valid scorecard plus a fitted head, never a set of weights
  proposed by the analyst. The existing add/reword/retire schema, feature
  budget, complete-answer requirement, propensity weighting, and out-of-fold
  calibration apply to every candidate.
- Share cached answers to identical questions across candidates. Batch newly
  proposed questions for the same item when the engine supports it. Price a
  batch before any live call.
- Predefine a few slices before seeing JEVPA's outcomes. Use slices with enough
  effective labels to evaluate; otherwise report them without letting a noisy
  slice decide the frontier. Record overall Brier, accuracy, calibration, and
  slice results. Counterfactual outcome checks on bios are constraints, not a
  performance gain that can be traded away.
- Keep the pool small and deterministic under a seed. A dominated candidate
  can be removed; a retained candidate must have a documented slice on which
  it earns its place. Recombination unions compatible questions, validates the
  resulting scorecard, and fits a new head. It never copies parent weights.
- The pool is research state. Only one evaluated scorecard is offered for human
  approval and serving. A no-change outcome is valid.

### Data separation

- **Discovery labels:** available to the analyst for reflection and to fit each
  candidate's head. Active picks retain their recorded selection propensities.
- **Selection labels:** withheld from the analyst and used to compare candidate
  scorecards and build the frontier, never to fit a candidate while searching.
  Repeated candidate search can overfit this set, so cap the search in advance
  and report its size and reuse. Once the winner is frozen, refit its head on
  discovery plus selection labels under the same fitting rules.
- **Final test:** never used for proposal, selection, stopping, or budget
  adjustment. Score the frozen winner once. The existing sentiment 600 and
  published bios tests have already informed this project's development; use
  them for historical comparison, not as a newly untouched confirmatory test.
  Register fresh, disjoint test items before running the comparison.

Start with the constructed sentiment task, where missing the subject-matter
axis is measurable. Then test on the Bias in Bios occupation task using its
real biographies and corpus occupation labels. State plainly that these are
scripted labels, not live reviewer feedback. Do not pool the two tasks into one
headline number.

## Work items

1. **Freeze the protocol.** Complete the live sentiment pre-registration before
   generating live proposals or answers. The exact split, request and analyst
   caps, slice rules, candidate cap, primary metric, accuracy tolerance,
   mechanism gate, and replication gate are in
   [`studies/JEVPA_LIVE_PROTOCOL.md`](studies/JEVPA_LIVE_PROTOCOL.md).
2. **Build a replayable search core.** Add a scorecard candidate record,
   per-slice scores, Pareto selection, question-set merge, and cost accounting.
   Put sentence-named specs beside each new module. Use a fake Jev client and
   scripted analyst in specs; no network or keys in `make test`.
3. **Run an offline pilot.** Replay recorded proposals and cached answers where
   possible. Use the local engine for any new exploratory questions. Check that
   all three arms use the same labels, candidate limits, and accounting. Treat
   this as a machinery check, not the confirmatory result.
4. **Run the registered comparison.** Price live requests before sending them.
   Log every proposal, accepted or rejected candidate, frontier change, merge,
   and final scorecard. Execute the matched-budget arms on the registered
   seeds and both tasks.
5. **Report the answer.** Publish paired held-out results by seed and task,
   Brier and accuracy, calibration, slice behavior, counterfactual outcomes,
   cost, and exact question wording. Explain failures and null results. Keep
   the current loop as the default unless the evidence supports changing it.

## Decision rule

The primary comparison is JEVPA versus batch plus greedy on fresh held-out
Brier, paired by seed within each task. Accuracy must not regress beyond the
pre-registered tolerance, and bios counterfactual constraints must pass. Report
whether retained candidates contributed complementary questions to the final
winner; a numerical gain without that mechanism is still useful but does not
support the Pareto hypothesis. Statistical uncertainty and all attempted runs
belong in the report.

The epic is done when the protocol, offline specs, costed experiment, and
reproducible report exist, regardless of whether JEVPA wins.
