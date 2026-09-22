# Jev spend log — bios_race2 study (full-name second attempt)

Every Jev request this study sends, logged before it is sent (`--price-only`), per the
pre-registration's money rule (`studies/PREREGISTERED.md`, "race from a full name, second
attempt"). `--price-only` reports **request counts and estimated input tokens**, not a dollar
figure -- neither `typesafe-sdk` 0.7.0 nor this repo's CLI exposes a price-per-request anywhere
(see `studies/bios_gender_spend.md`'s note, unchanged for this study). The pre-registration
fixes the count at exactly 500 bios x 16 versions = 8,000 requests, and this run's own rules
cap it at 8,500.

| when | step | items | requests priced | requests sent | notes |
|---|---|---:|---:|---:|---|
| 2026-09-22 | race2 fixtures build | 2,000 test items -> 1,968 kept, 32 excluded (no insertion point) | n/a | n/a | `scripts/build_bios_race2_fixtures.py`; wrote `fixtures/bios/race2_versions.jsonl`, 31,488 rows (1,968 x 16); wrote `fixtures/bios/race2_jev_subsample.txt`, 500 of the 1,968 eligible bios |
| 2026-09-22 | Jev subsample filter | 31,488 -> 8,000 rows (500-bio subsample x 16 versions) | n/a | n/a | `var/race2_jev_items.jsonl` (not committed -- reproducible from the two files above) |
| 2026-09-22 | Jev race2 answers (price check) | 8,000 | 8,000 | 0 | `--price-only`; matches the pre-registered 8,000 exactly, under the 8,500 cap |
