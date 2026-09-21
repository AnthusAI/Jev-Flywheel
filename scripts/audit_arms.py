#!/usr/bin/env python
"""Tally the analyst-arms study from its records, with the hand judgement beside the keyword screen.

    python scripts/audit_arms.py

``studies/arms.jsonl`` holds every run; ``studies/arms_judged.json`` says which of them proposed an
element about the text's subject matter, judged by reading. Gain is the last version's held-out
accuracy minus the best of the plain refits before it, on the same 600 items, paired within a run.
"""
import json
import statistics as st
from pathlib import Path

STUDIES = Path(__file__).resolve().parents[1] / "studies"


def gain(run):
    v = run["versions"]
    order = sorted(v, key=lambda k: int(k[1:]))
    fits = [v[k]["accuracy"] for k in order if v[k]["kind"] == "fit"]
    return (v[order[-1]]["accuracy"] - max(fits)) * 100


def main():
    runs = [json.loads(line) for line in (STUDIES / "arms.jsonl").read_text().splitlines()]
    judged = json.loads((STUDIES / "arms_judged.json").read_text())["named_the_axis"]
    found = {(j["arm"], j["model"], int(j["seed"])) for j in judged}
    print(f"{'arm':4}{'runs':>5}{'errors':>7}{'valid':>6}{'screen':>7}{'judged':>7}"
          f"{'promoted':>9}{'mean gain':>10}{'range':>13}{'gain | axis named':>19}")
    for arm in ("d0", "d1", "d2"):
        own = [r for r in runs if r.get("arm") == arm]
        errors = 12 - len(own)          # a failed run is recorded without an arm; 4 in all
        promoted = [r for r in own if r["decision"] == "promoted"]
        gains = [gain(r) for r in promoted]
        hits = [gain(r) for r in promoted if (arm, r["model"], int(r["seed"])) in found]
        print(f"{arm:4}{12:>5}{errors:>7}{len(own):>6}{sum(bool(r['found_the_plant']) for r in own):>7}"
              f"{sum((arm, r['model'], int(r['seed'])) in found for r in own):>7}{len(promoted):>9}"
              f"{st.mean(gains):>10.1f}{min(gains):>7.1f}..{max(gains):<5.1f}"
              f"{(f'{st.mean(hits):.1f} (n={len(hits)})' if hits else '-'):>19}")
    print(f"\nerror rows (no arm recorded): {sum(1 for r in runs if 'arm' not in r)}")


if __name__ == "__main__":
    main()
