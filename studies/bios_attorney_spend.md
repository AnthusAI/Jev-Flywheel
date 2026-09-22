# Jev spend log — bios_attorney study ("the learning loop on the pair that matters")

Every Jev request this study sends, logged before it is sent (`--price-only` first), per
`studies/PREREGISTERED.md`'s money rule. Section hard cap: 24,000 requests. Neither
`typesafe-sdk` nor `flywheel topup` exposes a dollar rate, so this log tracks requests and
input/output tokens, like every other bios spend log in this repo.

| when | step | items | requests priced | requests sent | notes |
|---|---|---:|---:|---:|---|
| 2026-09-22 | pool + changed held-out twins: priced | 2148 | 2148 | 0 | priced before sending; 2146 pool (attorney 2000, paralegal 146 -- paralegal train split exhausted by the existing held-out sample) + 2 twins whose text changed under the amended swap rule |
| 2026-09-22 | pool + changed held-out twins: sent | 2148 | 2148 | 2148 | 0 failures; 823,202 input tokens, 79,643 output tokens |
| 2026-09-22 | J1 seed 1: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round |
| 2026-09-22 | J1 seed 1: steering round (sent) | 140 | 140 | 140 | decision=promoted; 69,755 input / 9,137 output tokens so far |
| 2026-09-22 | J1 seed 1: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J1 seed 1: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J1 seed 2: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round |
| 2026-09-22 | J1 seed 2: steering round (sent) | 140 | 140 | 140 | decision=promoted; 69,935 input / 6,937 output tokens so far |
| 2026-09-22 | J1 seed 2: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J1 seed 2: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J1 seed 3: steering round (eval top-up) | 140 | 140 | 0 | priced as an upper bound before the round |
| 2026-09-22 | J1 seed 3: steering round (sent) | 140 | 140 | 140 | decision=rejected_by_metrics; 76,705 input / 14,000 output tokens so far |
| 2026-09-22 | J2 seed 1: steering round (eval top-up+ gate) | 140 | 280 | 0 | priced as an upper bound before the round |
| 2026-09-22 | J2 seed 1: steering round (sent) | 140 | 280 | 264 | decision=promoted; 103,068 input / 6,336 output tokens so far |
| 2026-09-22 | J2 seed 1: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J1 seed 1: serve held-out + twins | 4000 | 16 | 0 | priced before sending |
| 2026-09-22 | J1 seed 1: serve held-out + twins (sent) | 4000 | 16 | 0 | 16 failures |
| 2026-09-22 | J1 seed 2: serve held-out + twins | 4000 | 16 | 0 | priced before sending |
| 2026-09-22 | J1 seed 2: serve held-out + twins (sent) | 4000 | 16 | 0 | 16 failures |
| 2026-09-22 | J2 seed 1: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J2 seed 2: steering round (eval top-up+ gate) | 140 | 280 | 0 | priced as an upper bound before the round |
| 2026-09-22 | J2 seed 2: steering round (sent) | 140 | 280 | 264 | decision=promoted; 136,950 input / 18,744 output tokens so far |
| 2026-09-22 | J2 seed 2: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | J2 seed 2: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
| 2026-09-22 | J2 seed 3: steering round (eval top-up+ gate) | 140 | 280 | 0 | priced as an upper bound before the round |
| 2026-09-22 | J2 seed 3: steering round (sent) | 140 | 280 | 264 | decision=rejected_by_metrics; 131,327 input / 17,424 output tokens so far |
| 2026-09-22 | J2 seed 1: serve held-out + twins | 4000 | 16 | 0 | priced before sending |
| 2026-09-22 | J2 seed 1: serve held-out + twins (sent) | 4000 | 16 | 0 | 16 failures |
| 2026-09-22 | J2 seed 2: serve held-out + twins | 4000 | 16 | 0 | priced before sending |
| 2026-09-22 | J2 seed 2: serve held-out + twins (sent) | 4000 | 16 | 0 | 16 failures |
| 2026-09-22 | baseline-J1 seed 1: serve held-out + twins | 4000 | 4000 | 0 | priced before sending |
| 2026-09-22 | baseline-J1 seed 1: serve held-out + twins (sent) | 4000 | 4000 | 3984 | 16 failures |
