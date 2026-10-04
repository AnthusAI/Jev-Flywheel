# JEVPA offline pilot: search machinery, not discovery evidence

Protocol: [`JEVPA_PILOT_PROTOCOL.md`](JEVPA_PILOT_PROTOCOL.md). Reproducible
record: [`jevpa_pilot.json`](jevpa_pilot.json). Run from the repository root:

```bash
.venv/bin/python scripts/run_jevpa_offline.py --out studies/jevpa_pilot.json
```

This run uses the recorded 140 propensity-weighted sentiment labels for head
fitting, 500 disjoint pool items for candidate selection, and 600 further pool
items for an exploratory test (seed 20260924). All seven pre-existing candidate
questions have cached Jev answers on every item. There were no network, Jev,
or analyst calls.

| Arm | Candidate fits | Selected questions | Selection Brier | Exploratory test Brier | Test accuracy | Test ECE |
|---|---:|---|---:|---:|---:|---:|
| Frozen holistic v1 | 0 | none | — | 0.1541 | 0.7933 | 0.1186 |
| Greedy | 9 | praise, expectation | **0.1252** | **0.1107** | 0.8150 | 0.0388 |
| JEVPA | 8 | mixed, praise | 0.1340 | 0.1157 | **0.8350** | **0.0352** |

The positive/negative slice frontier retained the `praise` and `mixed`
singletons. Their one unique union was **also greedy's first merge**. Greedy
selected its other merge (`praise` plus `expectation`) on the 500 selection
items. JEVPA's pool had no second distinct union, so it left one of its nine
allowed fits unused. Its two-point higher test accuracy does not establish a
Pareto discovery: the question set was available to both arms, and JEVPA was
worse on the declared primary metric, Brier, by about 0.005.

This is evidence that the implementation can retain slice specialists, merge
their questions, refit rather than copy weights, and evaluate with complete
cached answers. It is **not** evidence that a reflective analyst discovers
better questions with Pareto search. The question bank was written before
JEVPA; the corpus labels are scripted; the corpus has been studied extensively;
and this is one split and one seed. The current flywheel should remain the
default. A next experiment needs multiple independently proposed questions
with answers on genuinely separate selection and test items, a budget-matched
greedy arm, and the protocol frozen before seeing those results. This pilot
does not trigger the planned Biased-Decisions extension.

During review, the runner's first implementation gave `expectation` one
feature instead of the three specified by the existing proposal rule. We
corrected it to use `apply_proposal` before recording the numbers above; the
initial exploratory numbers are not used in any conclusion. We also kept the
one-unused-fit limitation visible rather than introducing a new search rule
after seeing the first frontier.
