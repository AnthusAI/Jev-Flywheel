#!/usr/bin/env python
"""Build the race-name pools for the full-name counterfactual's second attempt.

    python scripts/build_name_pools.py

Reads four public-domain name tables (none of them committed to this repo -- see
``--rosenman-first``/``--rosenman-last``/``--ssa-names``/``--census-surnames`` below for where
to point this at a local copy) and writes ``fixtures/bios/name_pools.json``: for each of four
groups (white, black, hispanic, asian), a first-name pool split by gender and a last-name pool,
plus the provenance and thresholds used to build them. See
``studies/PREREGISTERED.md``, "race from a full name, second attempt", point 2.

Sources
-------
* **First-name race probability** and **last-name race probability**: Rosenman, Olivella and
  Imai (2023), *Scientific Data*, doi:10.7910/DVN/SGKW0K (CC0). Two tables, ``name,whi,bla,his,
  asi,oth`` -- each row a name and its estimated probability of being borne by someone of each
  census race group.
* **First-name gender**: SSA baby-names counts, 1970-2021 (Hugging Face mirror
  ``jbrazzy/baby_names``), summed by name and sex over the whole range.
* **Last-name frequency**: the Census Bureau's 2010 surname file (``Names_2010Census.csv``),
  the ``count`` column (total bearers, not race-specific -- race-specificity is what the
  Rosenman probability already supplies).

Thresholds, fixed in the pre-registration
------------------------------------------
* A first name enters a group's pool if the group's Rosenman probability is >= 0.8 *and* the
  SSA data gives it >= 20,000 births over 1970-2021 with >= 90% of those births one sex (that
  sex becomes the name's pool gender).
* A last name enters a group's pool if the group's Rosenman probability is >= 0.8 *and* the
  Census 2010 bearer count is >= 5,000.
* The Asian group gets no first-name pool at this threshold (0 names clear 0.8): fixture
  building falls back to the white first-name pool for Asian versions, paired with the Asian
  last-name pool, as stated in the pre-registration.

This script does not tune these thresholds to hit the pre-registration's expected pool sizes --
if the counts differ, the difference is recorded in the output, not chased away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

GROUPS = ("white", "black", "hispanic", "asian")
ROSENMAN_COL = {"white": "whi", "black": "bla", "hispanic": "his", "asian": "asi"}

FIRST_NAME_PROB = 0.8
FIRST_NAME_MIN_BIRTHS = 20_000
FIRST_NAME_MIN_GENDER_FRAC = 0.9
LAST_NAME_PROB = 0.8
LAST_NAME_MIN_BEARERS = 5_000

# The pre-registration's expected pool sizes (studies/PREREGISTERED.md, point 2), recorded here
# so this script can report any drift from them without tuning thresholds to match.
EXPECTED_SIZES = {
    "white": {"first_female": 190, "first_male": 174, "last": 3305},
    "black": {"first_female": 10, "first_male": 14, "last": 53},
    "hispanic": {"first_female": 11, "first_male": 39, "last": 267},
    "asian": {"first_female": 0, "first_male": 0, "last": 140},
}


def _first_name_genders(ssa_path: Path) -> Dict[str, str]:
    """Upper-cased first name -> "female"/"male" for names with enough total births and a
    strong-enough sex skew; names that don't clear both bars are absent (ambiguous or rare)."""
    ssa = pd.read_parquet(ssa_path)
    totals = ssa.groupby(["Names", "Sex"])["Count"].sum().unstack(fill_value=0)
    totals["total"] = totals.sum(axis=1)
    totals = totals[totals["total"] >= FIRST_NAME_MIN_BIRTHS]
    female = totals.get("F", 0)
    male = totals.get("M", 0)
    genders: Dict[str, str] = {}
    for name, total, f, m in zip(totals.index, totals["total"], female, male):
        if f / total >= FIRST_NAME_MIN_GENDER_FRAC:
            genders[str(name).upper()] = "female"
        elif m / total >= FIRST_NAME_MIN_GENDER_FRAC:
            genders[str(name).upper()] = "male"
    return genders


def _last_name_bearers(census_path: Path) -> Dict[str, int]:
    census = pd.read_csv(census_path)
    return dict(zip(census["name"].astype(str), census["count"].astype(int)))


def _title(name: str) -> str:
    return name.strip().title()


def build(rosenman_first: Path, rosenman_last: Path, ssa_names: Path, census_surnames: Path,
          out_path: Path) -> None:
    first_df = pd.read_csv(rosenman_first).dropna(subset=["name"])
    last_df = pd.read_csv(rosenman_last).dropna(subset=["name"]).drop_duplicates(subset=["name"])
    genders = _first_name_genders(ssa_names)
    bearers = _last_name_bearers(census_surnames)

    print(f"{len(first_df):,} first names, {len(last_df):,} last names in the Rosenman tables")
    print(f"{len(genders):,} SSA first names clear {FIRST_NAME_MIN_BIRTHS:,} births and "
          f"{FIRST_NAME_MIN_GENDER_FRAC:.0%} one sex")

    groups: Dict[str, Dict] = {}
    sizes: Dict[str, Dict[str, int]] = {}
    for group in GROUPS:
        col = ROSENMAN_COL[group]

        first_pool: Dict[str, List[str]] = {"female": [], "male": []}
        qualifying_first = first_df.loc[first_df[col] >= FIRST_NAME_PROB, "name"]
        for name in qualifying_first:
            gender = genders.get(name)
            if gender:
                first_pool[gender].append(_title(name))
        for gender in first_pool:
            first_pool[gender] = sorted(set(first_pool[gender]))

        qualifying_last = last_df.loc[last_df[col] >= LAST_NAME_PROB, "name"]
        last_pool = sorted({_title(name) for name in qualifying_last
                            if bearers.get(name, 0) >= LAST_NAME_MIN_BEARERS})

        uses_white_first = False
        if group == "asian" and not first_pool["female"] and not first_pool["male"]:
            uses_white_first = True

        groups[group] = {
            "first": first_pool,
            "last": last_pool,
            "uses_white_first_pool": uses_white_first,
        }
        sizes[group] = {"first_female": len(first_pool["female"]),
                        "first_male": len(first_pool["male"]), "last": len(last_pool)}

    print("\npool sizes (observed vs. pre-registered expectation):")
    deviations = []
    for group in GROUPS:
        obs = sizes[group]
        exp = EXPECTED_SIZES[group]
        line = (f"  {group:9s} first F {obs['first_female']:>4d} (exp {exp['first_female']:>4d})"
               f"  first M {obs['first_male']:>4d} (exp {exp['first_male']:>4d})"
               f"  last {obs['last']:>5d} (exp {exp['last']:>5d})")
        print(line)
        for key in ("first_female", "first_male", "last"):
            if obs[key] != exp[key]:
                deviations.append(f"{group}.{key}: observed {obs[key]}, pre-registered "
                                  f"{exp[key]} (difference {obs[key] - exp[key]:+d})")

    if deviations:
        print("\ndeviations from the pre-registered pool sizes (thresholds unchanged):")
        for line in deviations:
            print(f"  - {line}")
    else:
        print("\nall pool sizes match the pre-registration exactly.")

    payload = {
        "provenance": {
            "race_probability": ("Rosenman, Olivella & Imai (2023), Scientific Data, "
                                 "doi:10.7910/DVN/SGKW0K (CC0)"),
            "first_name_gender": ("SSA baby-names counts, 1970-2021 "
                                  "(Hugging Face mirror jbrazzy/baby_names)"),
            "last_name_frequency": "Census Bureau 2010 surname file (Names_2010Census.csv)",
        },
        "thresholds": {
            "first_name_prob": FIRST_NAME_PROB,
            "first_name_min_births": FIRST_NAME_MIN_BIRTHS,
            "first_name_min_gender_frac": FIRST_NAME_MIN_GENDER_FRAC,
            "last_name_prob": LAST_NAME_PROB,
            "last_name_min_bearers": LAST_NAME_MIN_BEARERS,
        },
        "pool_sizes": {"observed": sizes, "expected": EXPECTED_SIZES,
                       "deviations": deviations},
        "groups": groups,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rosenman-first", type=Path, required=True,
                        help="Rosenman/Olivella/Imai first-name race-probability table")
    parser.add_argument("--rosenman-last", type=Path, required=True,
                        help="Rosenman/Olivella/Imai last-name race-probability table")
    parser.add_argument("--ssa-names", type=Path, required=True,
                        help="SSA baby-names parquet (Names, Sex, Count, Year)")
    parser.add_argument("--census-surnames", type=Path, required=True,
                        help="Census 2010 surname CSV (name, count, ...)")
    parser.add_argument("--out", type=Path, default=Path("fixtures/bios/name_pools.json"))
    args = parser.parse_args()
    build(args.rosenman_first, args.rosenman_last, args.ssa_names, args.census_surnames,
          args.out)


if __name__ == "__main__":
    main()
