#!/usr/bin/env python
"""Reproduce the corpus claims in the README. Offline, no keys, no model.

Every number the README asserts about the *data* is computed here, so a reader can check them
rather than take them on trust:

    python scripts/audit_corpus.py

The keyword lists are ours, not the corpus's. They are a coarse proxy for "does this text carry
a sports/workplace cue" and they are deliberately simple so the result is easy to audit and hard
to tune. The Jev-based measurement of the same thing (which neutral items get a domain named, and how
accurate the rule is on those) reads the recording's cached `topic_domain` answers, so it also
runs offline. Those exist for 740 items: the 140 recorded labels and the 600 held-out sample.
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

    # How wrong the holistic answer is where a keyword rule sees a sports or an office cue.
    holistic = {}
    with gzip.open(FIX / "answers.jsonl.gz", "rt") as handle:
        for line in handle:
            r = json.loads(line)
            holistic[r["id"]] = r["answers"]["Sentiment"]["choice"]
    print("\nholistic answer wrong, by keyword cue (neutral tier / all items):")
    for name, words in (("sports cue", SPORTS), ("office cue", OFFICE)):
        hit = [r for r in rows if any(k in r["text"].lower() for k in words)
               and not (name == "office cue" and any(k in r["text"].lower() for k in SPORTS))]
        neutral = [r for r in hit if r["metadata"]["tier"] == "neutral"]
        wrong = lambda g: sum(holistic[r["id"]] != r["metadata"]["reference_label"] for r in g) / len(g)
        print(f"  {name}: {wrong(neutral):.0%} of {len(neutral)} / {wrong(hit):.0%} of {len(hit)}")
    recording_checks(rows)


def recording_checks(rows):
    """Claims about the recorded run, from the recording's own files."""
    rec = FIX / "recordings" / "simulated-labeler"
    tiers = ("strong", "medium", "weak", "neutral")
    meta = {r["id"]: r["metadata"] for r in rows}
    feedback = [json.loads(l) for l in (rec / "feedback.jsonl").read_text().splitlines()]
    pool = [m["tier"] for m in meta.values() if m["split"] == "pool"]
    got = [meta[f["item_id"]]["tier"] for f in feedback]
    print(f"\nlabel mix, {len(got)} recorded labels vs the {len(pool)}-item pool (strong/medium/weak/neutral):")
    print("  recorded " + "/".join(f"{got.count(t) / len(got):.1%}" for t in tiers))
    print("  pool     " + "/".join(f"{pool.count(t) / len(pool):.1%}" for t in tiers))
    topic = {}
    with gzip.open(rec / "extra_answers.jsonl.gz", "rt") as handle:
        for line in handle:
            r = json.loads(line)
            if r["name"] == "sentiment.topic_domain":
                topic[r["item_id"]] = r["answer"]["choice"]
    neutral = [i for i in topic if meta[i]["tier"] == "neutral"]
    named = [i for i in neutral if topic[i] != "something_else"]
    right = sum(("positive" if topic[i] == "sports_or_recreation" else "negative")
                == meta[i]["reference_label"] for i in named)
    print(f"neutral items with a topic answer: {len(neutral)}; a domain named on {len(named)} "
          f"({len(named) / len(neutral):.0%}); sports->positive, workplace->negative right on "
          f"{right / len(named):.1%}")
    by = {}
    for f in feedback:
        c = by.setdefault(topic.get(f["item_id"], "?"), [0, 0])
        c[0] += 1
        c[1] += not f["is_agreement"]
    print("recorded labels the holistic answer got wrong, by Jev's topic answer: "
          + "  ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
