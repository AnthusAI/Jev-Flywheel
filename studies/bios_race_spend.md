# Jev spend log — bios_race study

Every Jev request this study sends, logged before it is sent (`--price-only`, or the
equivalent plan check for a script that calls `JevSession` directly), per the
pre-registration's money rule (`studies/PREREGISTERED.md`, "does the engine read race from a
name?"). `--price-only` reports **request counts and estimated input tokens**, not a dollar
figure — neither `typesafe-sdk` 0.7.0 nor this repo's CLI exposes a price-per-request
anywhere (see `studies/bios_gender_spend.md`'s note, unchanged for this study). The
pre-registration fixes the count at exactly 4,713 requests (3 name versions x 1,571 bios with
a subject pronoun) and caps any run at 5,000.

| when | step | items | requests priced | requests sent | notes |
|---|---|---:|---:|---:|---|
| 2026-09-22 | race fixtures build | 2,000 test items -> 1,571 kept, 429 excluded (no subject pronoun) | n/a | n/a | `scripts/build_bios_race_fixtures.py`; wrote `fixtures/bios/race_versions.jsonl`, 4,713 rows (1,571 x 3), matching the pre-registration exactly |
| 2026-09-22 | Laya race answers | 4,713 | 0 (free, local) | 4,713 | `fixtures/bios/answers-race-laya.jsonl.gz`; ~17-23 items/s, 0 failures |
| 2026-09-22 | Jev race answers (price check) | 4,713 | 4,713 | 0 | `--price-only`; matches the pre-registered 4,713 exactly, under the 5,000 cap |
| 2026-09-22 | Jev race answers (send) | 4,713 | 4,713 | 4,713 | 1,838,465 input tokens, 176,450 output tokens, 0 failures. `fixtures/bios/answers-race.jsonl.gz` |
