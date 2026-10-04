# JEVPA live sentiment comparison protocol

Freeze this protocol before generating any live proposals or Jev answers for this
comparison. The question is whether Pareto retention and recombination improve
held-out decisions over greedy selection when both receive the same live proposal
and answer budget. Labels are scripted corpus labels, not live human feedback.

## Data and split

- Use the existing 140 propensity-recorded sentiment labels as discovery data for
  fitting every candidate. Both arms see the same discovery items and labels.
- Exclude those 140 IDs and every item used in the offline pilot's 500-item
  selection and 600-item test sets. Sort the remaining pool IDs, shuffle with
  seed **20260925**, and take the first **300** as the live selection set and
  the next **500** as the final test set. Freeze and record the IDs before any
  live calls.
- The analyst may see discovery data only. It must not see selection or test
  labels. Selection labels may rank candidates and form the JEVPA frontier, but
  may not fit a candidate. Test labels stay sealed until both winning scorecards
  and all fitted parameters are frozen; score each winner once.
- If the pool cannot supply all 800 disjoint items after exclusions, do not
  substitute pilot items. Record the shortfall and do not call the run
  confirmatory.

## Arms and matched budget

- **Greedy** and **JEVPA** receive the same four initial analyst calls, with
  identical prompts, discovery evidence, and outputs. Freeze those four
  questions and share their Jev answers across both arms.
- Each arm then gets two branch analyst calls, for at most **8 analyst calls
  total**. Use the same model, prompt template, discovery evidence, and output
  limits. Record complete prompts and replies. Greedy retains its best overall
  candidate; JEVPA may retain a Pareto pool capped at **4**.
- Ask all four shared initial questions in one batch over the 440 discovery and
  selection items. Ask each arm's two branch questions together in one batch
  over those same 440 items. Ask the frozen winners' questions in one final
  batch over the 500 test items, reusing answers already cached for identical
  questions. The ceiling is **1,820 item-batch Jev requests** (440 + 880 + 500),
  with no additional Jev requests. Price and record each batch before sending;
  stop if the request ceiling or a provider cost ceiling is reached.
- Both arms use the same candidate cap of **9 fits**: four initial candidates
  and up to five branch/merge candidates. Refit each candidate from the same
  140 trusted discovery labels under the existing propensity, complete-feature,
  and out-of-fold calibration rules. Never copy parent weights. Report actual
  fits, failed fits, unique questions, cached answer reuse, and request counts.
- JEVPA slice scores use positive and negative selection-label slices, each
  requiring at least **50 items**. If either slice is too small, report it and
  do not use it to award Pareto status. Use the existing deterministic
  dominance and compatible-union rules; do not relax them after seeing results.
- An unanswered item or failed candidate is reported as such. Do not silently
  score a smaller sample; the paired primary comparison requires complete
  coverage of all 500 test items in both arms.

## Outcomes and decision gates

- **Primary metric:** paired test-set Brier difference, `JEVPA − greedy`; lower
  is better. Report the difference and both arm values. A result is practically
  positive only if JEVPA improves Brier by at least **0.005** (difference at or
  below −0.005).
- **Accuracy guardrail:** JEVPA test accuracy must be no more than **0.02** below
  greedy. Also report accuracy, ECE, calibration details, positive/negative
  slice Brier, exact selected question keys and wording, and full candidate
  lineage.
- **Mechanism gate:** a Pareto-mechanism result requires a final JEVPA-only
  candidate that is a compatible merge of distinct candidates retained for
  different qualifying slices. The same question set must not have been
  evaluated by greedy. If the frontier has no distinct slice specialists or
  yields no such unique merge, stop the mechanism claim as **no opportunity**;
  report available outcomes as exploratory and do not spend the remaining
  budget inventing a fallback.
- **Single-run conclusion:** support for a Pareto advantage requires all three:
  the mechanism gate passes, the primary Brier improvement reaches 0.005, and
  the accuracy guardrail passes. Otherwise report a null, adverse, or
  mechanism-inconclusive result as appropriate. Do not change the current
  flywheel default from this single run.
- **Replication gate:** any claim that JEVPA should become the default requires
  a separately preregistered replication on new, disjoint selection and test
  items, with the same budgets and thresholds, that independently passes all
  three gates. Do not pool the first run and replication to rescue a failed
  replication. The Biased-Decisions extension is not triggered by this study.

## Required report

Publish the frozen split IDs or their checksums, seed, prompts and replies,
question wording, candidate and frontier/merge lineage, fit status, answer
coverage, item-batch request and provider-cost totals, paired test Brier and
accuracy, ECE/calibration, slice results, all gate outcomes, and deviations.
Report every attempted run. State explicitly that this study uses scripted
labels and does not establish performance on live human feedback.
