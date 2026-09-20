#!/usr/bin/env python
"""Reproduce the corpus claims in the README. Offline, no keys, no model.

Every number the README asserts about the *data* is computed here, so a reader can check them
rather than take them on trust:

    python scripts/audit_corpus.py

The keyword lists are ours, not the corpus's. They are a coarse proxy for "does this text carry
a sports/workplace cue" and they are deliberately simple so the result is easy to audit and hard
to tune. The Jev-based measurement of the same thing (57% of neutral items get a domain named,
and the rule is 90.3% accurate on those) needs an API key and lives in the README's prose.
"""
import gzip
import json
import statistics as st
from pathlib import Path

FIX = Path(__file__).resolve().parents[1] / "fixtures"
SPORTS = ("practice", "team", "coach", "athlet", "game", "match", "training", "swim", "golf",
          "tennis", "row", "box", "player", "tournament", "season", "field", "gym", "skat",
          "baseball", "track", "soccer", "run")
OFFICE = ("meeting", "office", "employee", "timesheet", "printer", "report", "deadline",
          "department", "manager", "conference", "email", "memo", "staff", "document",
          "schedul", "policy")


def main():
    rows = [json.loads(l) for l in (FIX / "items.jsonl").read_text().splitlines()]
    print(f"corpus: {len(rows)} items")
    by_split = {}
    for r in rows:
        by_split[r["metadata"]["split"]] = by_split.get(r["metadata"]["split"], 0) + 1
    print(f"splits: {by_split}\n")

    print(f"{'tier':10}{'label':10}{'n':>6}{'sports-ish':>12}{'office-ish':>12}")
    for tier in ("strong", "medium", "weak", "neutral"):
        for label in ("positive", "negative"):
            g = [r for r in rows if r["metadata"]["tier"] == tier
                 and r["metadata"]["reference_label"] == label]
            s = sum(any(k in r["text"].lower() for k in SPORTS) for r in g)
            o = sum(any(k in r["text"].lower() for k in OFFICE) for r in g)
            print(f"{tier:10}{label:10}{len(g):>6}{s/len(g):>11.0%}{o/len(g):>12.0%}")

    # Deduplicated by text when the fixtures were built, and it fell unevenly: the shipped
    # corpus is not balanced, which matters when reading an accuracy against a 50% intuition.
    labels = {}
    for r in rows:
        key = r["metadata"]["reference_label"]
        labels[key] = labels.get(key, 0) + 1
    total = sum(labels.values())
    print("\nclass balance after de-duplication: "
          + "  ".join(f"{k} {v} ({v / total:.0%})" for k, v in sorted(labels.items()))
          + f"   -> a majority-class baseline scores {max(labels.values()) / total:.1%}")

    texts = {r["id"]: r["text"] for r in rows}
    xs, ys = [], []
    with gzip.open(FIX / "answers.jsonl.gz", "rt") as handle:
        for line in handle:
            r = json.loads(line)
            used = (r.get("usage") or {}).get("input_tokens")
            if used and r["id"] in texts:
                xs.append(len(texts[r["id"]]))
                ys.append(used)
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    slope = (sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n) / st.pvariance(xs)
    ordered = [y for _, y in sorted(zip(xs, ys))]
    print(f"\nrequest cost over {n} items, eight questions each:")
    print(f"  mean {st.mean(ys):.0f} input tokens   shortest tenth {st.mean(ordered[:n // 10]):.0f}"
          f"   longest tenth {st.mean(ordered[-n // 10:]):.0f}")
    print(f"  fitted: tokens = {slope:.3f} x characters + {my - slope * mx:.0f}"
          "   (per-request overhead dominates on single-sentence texts)")


if __name__ == "__main__":
    main()
