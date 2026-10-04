# JEVPA offline pilot protocol

Written before running the pilot outcome. This checks the search machinery on
the existing sentiment fixture. It is **exploratory**: these are scripted corpus
labels, the seven candidate questions were written before JEVPA, and the corpus
has already informed this project's development.

## Inputs and split

- Use the 140 recorded feedback labels, with their recorded selection
  propensities, as discovery data for every candidate head fit.
- Candidate questions are the seven elements of
  `fixtures/scorecards/reference_full.yaml`, with exactly their cached wording.
  Each candidate keeps the holistic question and adds one of those elements.
- Sort the IDs of the corpus `pool` split after removing the 140 discovery IDs,
  then shuffle with seed **20260924**. The first **500** are selection items;
  the next **600** are an exploratory test. Use their reference labels, with no
  analyst or proposal process shown either split.
- The recorded `topic_domain` question has no answers on these 1,100 pool
  items, so it is excluded from this comparison. Its published historical
  result may be displayed separately, never mixed into these arms.

## Arms and budget

- **Frozen incumbent:** the v1 holistic-only scorecard, reported as context.
- **Greedy:** fit seven singleton candidate scorecards. Rank them by overall
  selection Brier. Spend two further candidate fits on two-question sets made
  from the best singleton and the next two ranked distinct singletons.
- **JEVPA:** fit the same seven singletons. Use selection Brier on the positive
  and negative reference-label slices, requiring at least 50 items per slice.
  Keep a Pareto pool capped at four candidates. Spend two further fits on compatible unions of
  distinct slice-winning candidates, with deterministic fallback to the next
  Pareto candidate when the winners are the same. Refit every union from the
  140 discovery labels; do not combine parent weights.

Each search arm may fit **up to nine** candidates and may use only the same eight
cached Jev answers per item (holistic plus seven elements). There are no new
Jev requests or analyst calls in this pilot. Report actual candidate fits and
question-answer coverage alongside results. A candidate that cannot fit or
lacks a complete answer is a recorded failure, not a silently smaller sample.

## Selection and report

Select each arm's winner by overall selection Brier among its evaluated
candidates. Score its frozen fitted version once on the 600 exploratory test
items. Report Brier, accuracy, ECE, scorecard question keys, selection slices,
the Pareto pool, and whether a merge differs from greedy. The primary pilot
comparison is JEVPA versus greedy on the same selection and test items. This
pilot can demonstrate that search and merging run end to end; it cannot support
a claim that Pareto improves metacognitive discovery or generalizes to live
human feedback.

## Implementation note, recorded after the first machinery run

The singleton Pareto frontier contained only two distinct candidates, so it
offered one unique two-question union. JEVPA therefore used eight candidate
fits while greedy used nine. We did **not** add a dominated candidate or repeat
an identical fit to exhaust the budget after seeing this. The result is an
equal *maximum* budget comparison with unequal realized fit counts; the report
must display that limitation.
