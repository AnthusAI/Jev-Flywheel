# Jev spend log — bios_gender study

Every Jev request this study sends, logged before it is sent (`flywheel topup` without
`--yes`, or the equivalent plan check for a script that calls `JevSession` directly), per
the pre-registration's money rule. `flywheel topup`/`workspace.cache.plan` report **request
counts and estimated input tokens**, not a dollar figure — neither the CLI nor
`typesafe-sdk` 0.7.0 exposes a price-per-request anywhere in this repo. Each line below
records what was priced and what was actually sent; a running total in requests stands in
for the $60 cap until a $/request rate is available (see the step-2 note below).

| when | step | items | requests priced | requests sent | notes |
|---|---|---:|---:|---:|---|
| 2026-09-22 | credential check | 1 (scratch workspace, `var/bios-cred-check`, empty answer cache) | 1 | 1 | `TYPESAFE_API_KEY` confirmed working; 356 input tokens, 0 failures |
| 2026-09-22 | J0 test batch | 100 | 100 | 100 | confirms the pipeline works; 38,723 input tokens (387/item), 3,769 output tokens; 0 failures |
| 2026-09-22 | J0 (build `fixtures/bios/answers.jsonl.gz`) | 8,000 (all pool + test + counterfactual twins) | 7,900 remaining after the test batch | 7,900 | 3,018,775 input tokens, 295,352 output tokens, 0 failures. Total for J0 including the test batch: 8,000 requests, ~3.06M input / ~299K output tokens |
| 2026-09-22 | J1 seed 1: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J1 seed 1: steering round (sent) | 140 | 140 | 52 | decision=not_evaluable; 23,894 input / 2,600 output tokens so far |
| 2026-09-22 | J1 seed 1: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J1 seed 1: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J1 seed 1: steering round (sent) | 140 | 140 | 140 | decision=promoted; 72,053 input / 7,000 output tokens so far |
| 2026-09-22 | J1 seed 1: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J1 seed 1: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J1 seed 2: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J1 seed 2: steering round (sent) | 140 | 140 | 140 | decision=promoted; 67,961 input / 10,080 output tokens so far |
| 2026-09-22 | J1 seed 2: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J1 seed 2: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J1 seed 3: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J1 seed 3: steering round (sent) | 140 | 140 | 140 | decision=promoted; 67,695 input / 9,800 output tokens so far |
| 2026-09-22 | J1 seed 3: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J1 seed 3: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J2 seed 1: steering round (eval top-up+ gate) | 140 | 280 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J2 seed 1: steering round (sent) | 140 | 280 | 264 | decision=rejected_by_metrics; 141,699 input / 13,464 output tokens so far |
| 2026-09-22 | J2 seed 2: steering round (eval top-up+ gate) | 140 | 280 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J2 seed 2: steering round (sent) | 140 | 280 | 264 | decision=rejected_by_metrics; 140,730 input / 18,480 output tokens so far |
| 2026-09-22 | J2 seed 3: steering round (eval top-up+ gate) | 140 | 280 | 0 | priced as an upper bound before the round (the exact top-up depends on whether an element is proposed) |
| 2026-09-22 | J2 seed 3: steering round (sent) | 140 | 280 | 264 | decision=promoted; 139,906 input / 19,800 output tokens so far |
| 2026-09-22 | J2 seed 3: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J2 seed 3: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
