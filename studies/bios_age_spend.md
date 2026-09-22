# Jev spend log — bios_age study

Every Jev request this study sends, logged before it is sent (`--price-only`), per the
pre-registration's money rule (`studies/PREREGISTERED.md`, "does the engine read age?"). As
with the race study, `--price-only` reports **request counts**, not a dollar figure — neither
`typesafe-sdk` 0.7.0 nor this repo's CLI exposes a price-per-request anywhere. The
pre-registration fixes the count at exactly 4,924 requests (4 ages x 1,231 eligible bios) and
caps any run at 5,200.

| when | step | items | requests priced | requests sent | notes |
|---|---|---:|---:|---:|---|
| 2026-09-22 | age fixtures build | 2,000 test items -> 1,231 eligible (640 surgeon, 591 physician), 769 excluded (no subject pronoun, a year before 2000, or a 10+ year duration) | n/a | n/a | `scripts/build_bios_age_fixtures.py`; wrote `fixtures/bios/age_versions.jsonl`, 4,924 rows (1,231 x 4), matching the pre-registration exactly |
| 2026-09-22 | Jev age answers (price check) | 4,924 | 4,924 | 0 | `--price-only`; matches the pre-registered 4,924 exactly, under the 5,200 cap |
| 2026-09-22 | Jev age answers (send) | 4,924 | 4,924 | 4,924 | 1,925,820 input tokens, 184,228 output tokens, 0 failures. `fixtures/bios/answers-age.jsonl.gz` |
| 2026-09-22 | Laya age answers | 4,924 | 0 (free, local) | 4,924 | `fixtures/bios/answers-age-laya.jsonl.gz`; 0 failures |
