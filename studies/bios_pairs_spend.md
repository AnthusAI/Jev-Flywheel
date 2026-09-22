# Jev spend log — bios_pairs study

Every Jev request this study sends, logged before it is sent (`--price-only`), per the
pre-registration's money rule (`studies/PREREGISTERED.md`, "does the gender result hold on
other decisions?"). `--price-only` reports **request counts and estimated input tokens**, not a
dollar figure — neither `typesafe-sdk` 0.7.0 nor this repo's CLI exposes a price-per-request
anywhere (see `studies/bios_gender_spend.md`'s note, unchanged for this study). The
pre-registration fixes the count at exactly 12,000 requests (3 pairs x 4,000 answers per pair:
1,000 per label, plus each item's counterfactual twin) and caps any run at 12,600.

| when | step | items | requests priced | requests sent | notes |
|---|---|---:|---:|---:|---|
| 2026-09-22 | pairs fixtures build | 3 pairs, 1,000 per label from the train split, seed 0 | n/a | n/a | `scripts/build_bios_pairs_fixtures.py`; paralegal's train-split pool had >=1,000 rows, so all three pairs got the full 1,000/1,000, no deviation |
| 2026-09-22 | Jev, nurse/physician (price check) | 4,000 | 4,000 | 0 | `--price-only`; matches the pre-registered 4,000 for this pair |
| 2026-09-22 | Jev, nurse/physician (send) | 4,000 | 4,000 | 4,000 | 1,526,491 input tokens, 144,000 output tokens, 0 failures. `fixtures/bios_pairs/nurse_physician/answers.jsonl.gz` |
| 2026-09-22 | Jev, paralegal/attorney (price check) | 4,000 | 4,000 | 0 | `--price-only` |
| 2026-09-22 | Jev, paralegal/attorney (send) | 4,000 | 4,000 | 4,000 | 1,529,881 input tokens, 149,568 output tokens, 0 failures. `fixtures/bios_pairs/paralegal_attorney/answers.jsonl.gz` |
| 2026-09-22 | Jev, teacher/professor (price check) | 4,000 | 4,000 | 0 | `--price-only` |
| 2026-09-22 | Jev, teacher/professor (send) | 4,000 | 4,000 | 4,000 | 1,541,483 input tokens, 143,666 output tokens, 0 failures. `fixtures/bios_pairs/teacher_professor/answers.jsonl.gz` |
| 2026-09-22 | Laya, all three pairs | 12,000 | 0 (free, local) | 12,000 | `fixtures/bios_pairs/<pair>/answers-laya.jsonl.gz` |

**Total: 12,000 Jev requests (the exact pre-registered count), 4,597,855 input tokens, 437,234
output tokens, 0 failures — under the 12,600 cap.** Laya's 12,000 answers were free and local.
