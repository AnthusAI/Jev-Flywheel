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
