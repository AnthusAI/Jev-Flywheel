#!/usr/bin/env python
"""How does the head do as labels grow, with and without the discovered element?

    python scripts/learning_curve.py --scratch /tmp/lc --prepare   # once: replay + Laya top-up
    python scripts/learning_curve.py --scratch /tmp/lc             # the curves

Offline and free. The simulated labeler is the corpus's own reference label, so any pool
item can be "labeled"; this script draws random pool subsets of growing size (uniform
propensity, so no inverse-propensity weighting is in play), fits the head with the repo's own
``fit_head`` / ``with_fit``, and scores every fitted scorecard on fixed held-out sets:

* ``paper600``  the README's 600 held-out items (the only test items with Jev answers to
                the discovered ``topic_domain`` element)
* ``full``      all 3,521 held-out items (usable wherever the answers exist)

Four question sets, all answerable from the committed fixtures plus one free local Laya pass:

    H      the holistic sentiment answer alone (the recorded run's v1..v3)
    H+T    holistic + the discovered ``topic_domain`` element (the recorded run's v4)
    H+7    holistic + the seven reference elements cached in fixtures/answers*.jsonl.gz
    H+7+T  everything

Jev has ``topic_domain`` answers only for the 140 recorded labeled items and the 600 test
items, so on Jev the ``+T`` curves stop at 140 labels. Laya answers the element locally, so
``--prepare`` asks Laya about every item once (about 8,801 forward passes) and the Laya
curves run to every pool item. A hybrid row (Jev holistic + Laya topic) is included because
the topic question is nearly free locally; it is labeled as such.

Nothing here is fitted on a held-out item. Results go to one JSONL, one row per fit.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_flywheel.answers import AnswerCache  # noqa: E402
from jev_flywheel.cli import PACKAGED_FIXTURES  # noqa: E402
from jev_flywheel.evaluate import summarize  # noqa: E402
from jev_flywheel.fit import build_training_set, fit_head, with_fit  # noqa: E402
from jev_flywheel.items import FeedbackItem, JsonlStore, agrees  # noqa: E402
from jev_flywheel.ladder import LadderRefusal  # noqa: E402
from jev_flywheel.proposal import default_features  # noqa: E402
from jev_flywheel.recording import replay  # noqa: E402
from jev_flywheel.report import complete_items  # noqa: E402
from jev_flywheel.scorecard import Scorecard  # noqa: E402
from jev_flywheel.scoring import predict  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

RECORDING = PACKAGED_FIXTURES / "recordings" / "simulated-labeler"
SCORE = "Sentiment"
SIZES = (50, 100, 140, 200, 300, 500, 1000, 2000, 5280)
SEEDS_FOR = lambda n: 5 if n <= 500 else (3 if n <= 2000 else 1)  # noqa: E731
PREFIXES = (37, 52, 87, 100, 120, 140)


# ---- workspaces ------------------------------------------------------------------------

def prepare(scratch: Path) -> None:
    jev_dir = scratch / "jev" / "var"
    if not (jev_dir / "scorecards" / "v4.yaml").exists():
        print("replaying the recording against Jev ...")
        replay(RECORDING, jev_dir, PACKAGED_FIXTURES)
    jev = Workspace(jev_dir)
    topic_q = {k: v for k, v in jev.scorecard(4).questions().items() if k != SCORE}
    assert list(topic_q) == ["sentiment.topic_domain"], topic_q

    laya_dir = scratch / "laya" / "var"
    laya = Workspace(laya_dir)
    if not laya.exists:
        print("building a Laya workspace from fixtures/answers-laya.jsonl.gz ...")
        laya = Workspace.init(laya_dir, PACKAGED_FIXTURES, answers="answers-laya.jsonl.gz",
                              engine="laya")
    plan = laya.cache.plan([i.id for i in laya.items], topic_q)
    print(f"Laya topic_domain answers missing for {plan.requests} of {len(laya.items)} items")
    if plan.requests:
        from jev_flywheel.jev import JevSession
        from jev_flywheel.laya import LayaClient
        client = LayaClient()
        client.warm()
        started = time.time()
        report = asyncio.run(laya.cache.fill(
            JevSession(client_factory=lambda: client), laya.items, topic_q, concurrency=1))
        print(f"asked Laya {report.requested} times, {report.failures} failed, "
              f"{time.time() - started:.0f}s")


# ---- scorecards for each question set ----------------------------------------------------

def make_card(v4: Scorecard, reference: Scorecard, *, topic: bool, seven: bool) -> Scorecard:
    """A one-score card with the chosen elements and the repo's default feature per element."""
    config = v4.to_config()
    score = config["scores"][0]
    topic_elements = [e for e in score.get("elements", []) if e["key"] == "topic_domain"]
    seven_elements = reference.to_config()["scores"][0]["elements"]
    elements = (seven_elements if seven else []) + (topic_elements if topic else [])
    features = ["self.holistic.clr.positive"]
    for e in elements:
        features += default_features(e["key"], e["question_type"], e.get("criteria"))
    score["elements"] = elements
    score["decision"] = {
        "model": "multinomial_logistic", "classes": ["positive", "negative"],
        "features": features, "parameters": {"weights": {}},
    }
    return Scorecard.from_config(config)


CONFIGS = {
    "H": dict(topic=False, seven=False),
    "H+T": dict(topic=True, seven=False),
    "H+7": dict(topic=False, seven=True),
    "H+7+T": dict(topic=True, seven=True),
}


# ---- labels ------------------------------------------------------------------------------

def simulated_feedback(items, ids: List[str]) -> List[FeedbackItem]:
    """The simulated labeler on a chosen set of pool items, drawn uniformly (propensity 1)."""
    return [FeedbackItem(id=f"lc-{i}", item_id=i, score_name=SCORE,
                         initial_answer_value=None, final_answer_value=items[i].reference_label,
                         is_agreement=None, metadata={"propensity": 1.0}) for i in ids]


def recorded_feedback() -> List[FeedbackItem]:
    return JsonlStore(RECORDING / "feedback.jsonl", FeedbackItem).all()


# ---- scoring -----------------------------------------------------------------------------

def score_on(card: Scorecard, cache: AnswerCache, items, ids) -> Optional[dict]:
    score = card.score(SCORE)
    questions = card.questions()
    answers = cache.bulk_partial_answers(ids, questions)
    wanted = set(questions)
    confidences, correct, tiers = [], [], collections.defaultdict(list)
    complete = 0
    for item_id in ids:
        item = items[item_id]
        complete += wanted <= set(answers[item_id])
        result = predict(score, answers[item_id])
        hit = int(agrees(result.value, item.reference_label))
        confidences.append(result.confidence or 0.0)
        correct.append(hit)
        tiers[item.metadata["tier"]].append(hit)
    s = summarize(confidences, correct)
    return {"n": s.n, "accuracy": round(s.accuracy, 4), "ece": round(s.ece, 4),
            "brier": round(s.brier, 4), "coverage": round(complete / len(ids), 4),
            "by_tier": {t: round(sum(v) / len(v), 4) for t, v in sorted(tiers.items())}}


def one_fit(card: Scorecard, cache: AnswerCache, feedback: List[FeedbackItem]):
    score = card.score(SCORE)
    training = build_training_set(score, card.questions(), cache, feedback)
    if training.needs_answers:
        return None, {"error": f"{len(training.needs_answers)} labeled items lack answers"}
    try:
        result = fit_head(training, score)
    except LadderRefusal as refusal:
        return None, {"error": f"refused: {refusal}"}
    if not result.fitted:
        return None, {"error": f"held: {result.reason}"}
    fitted = with_fit(card, SCORE, result)
    return fitted, {"tier": result.tier.name, "n_train": result.n,
                    "n_effective": round(result.n_effective, 1), "C": result.chosen_c,
                    "oof_accuracy": round(result.metrics.accuracy, 4),
                    "oof_brier": round(result.metrics.brier, 4),
                    "weights": {cls: {k: round(v, 3) for k, v in w.items()}
                                for cls, w in result.head["weights"].items()}}


# ---- the study ---------------------------------------------------------------------------

def main(scratch: Path, out: Path, engines: List[str], residuals: bool) -> None:
    jev = Workspace(scratch / "jev" / "var").require()
    laya = Workspace(scratch / "laya" / "var").require()
    items = {i.id: i for i in jev.items}
    pool = [i.id for i in jev.split("pool")]
    test = [i.id for i in jev.split("test")]
    paper600 = sorted(complete_items(jev, "test"))
    assert len(paper600) == 600
    recorded = recorded_feedback()
    recorded_ids = [f.item_id for f in recorded]
    v4 = jev.scorecard(4)
    reference = Scorecard.from_yaml((PACKAGED_FIXTURES / "scorecards" / "reference_full.yaml").read_text())
    cards = {name: make_card(v4, reference, **spec) for name, spec in CONFIGS.items()}

    # A hybrid cache: Jev's answers for everything, plus Laya's answers to the topic element.
    topic_q = {k: v for k, v in v4.questions().items() if k != SCORE}
    hybrid = AnswerCache()
    for item_id, name, qhash, answer in jev.cache.rows():
        hybrid.put_hashed(item_id, name, qhash, answer)
    for item_id, name, qhash, answer in laya.cache.rows():
        if name in topic_q:
            hybrid.put_hashed(item_id, name, qhash, answer)  # overrides Jev's 740 topic rows

    caches = {"jev": jev.cache, "laya": laya.cache, "jev+laya-topic": hybrid}
    rows: List[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("weights", "by_tier_paper600",
                                                                     "by_tier_full")}))

    def evaluate(engine: str, config: str, label_source: str, n: int, seed: int,
                 feedback: List[FeedbackItem]) -> None:
        cache = caches[engine]
        fitted, info = one_fit(cards[config], cache, feedback)
        row = {"engine": engine, "config": config, "labels": label_source, "n": n, "seed": seed,
               **info}
        if fitted is not None:
            p = score_on(fitted, cache, items, paper600)
            row.update(paper600=p["accuracy"], ece600=p["ece"], brier600=p["brier"],
                       coverage600=p["coverage"], by_tier_paper600=p["by_tier"])
            f = score_on(fitted, cache, items, test)
            # Only meaningful where every held-out item has every answer.
            if f["coverage"] == 1.0:
                row.update(full=f["accuracy"], ece_full=f["ece"], brier_full=f["brier"],
                           by_tier_full=f["by_tier"])
        emit(row)

    for engine in engines:
        for config in CONFIGS:
            jev_topic = engine.startswith("jev") and "+T" in config and engine == "jev"
            for n in SIZES:
                if jev_topic and n > 140:
                    continue          # Jev has topic answers for the 140 recorded items only
                for seed in range(SEEDS_FOR(n)):
                    rng = random.Random(1000 * seed + n)
                    universe = recorded_ids if jev_topic else pool
                    ids = universe if n >= len(universe) else rng.sample(universe, n)
                    evaluate(engine, config, "random-pool" if not jev_topic else "random-of-recorded",
                             len(ids), seed, simulated_feedback(items, ids))
            # The recorded run's own labels, in order, with their propensities (IPW on).
            if engine in ("jev", "laya"):
                for k in PREFIXES:
                    evaluate(engine, config, "recorded-prefix", k, 0, recorded[:k])

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} rows to {out}")
    summarize_rows(rows)
    if residuals:
        residual_analysis(items, cards, caches, paper600, test, recorded, pool)


def summarize_rows(rows: List[dict]) -> None:
    print("\nmean held-out accuracy over seeds (paper600 / full) by engine, config, n:")
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("labels") in ("random-pool", "random-of-recorded") and "paper600" in r:
            groups[(r["engine"], r["config"], r["n"])].append(r)
    last = None
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        if last != key[:2]:
            print(f"\n  {key[0]:14} {key[1]:6}")
            last = key[:2]
        p = [r["paper600"] for r in g]
        f = [r["full"] for r in g if "full" in r]
        e = [r["ece600"] for r in g]
        print(f"    n={key[2]:5d}  paper600 {statistics.mean(p):.3f}"
              f"{'' if len(p) < 2 else f' ±{statistics.stdev(p):.3f}'}"
              f"   ece {statistics.mean(e):.3f}"
              + (f"   full {statistics.mean(f):.3f}" if f else "")
              + f"   oof {statistics.mean(r['oof_accuracy'] for r in g):.3f}   ({len(p)} seeds)")
    print("\nrecorded-prefix (the run's own actively selected labels, IPW-weighted):")
    for r in rows:
        if r.get("labels") == "recorded-prefix":
            print(f"  {r['engine']:6} {r['config']:6} k={r['n']:3d}  "
                  + (f"paper600 {r['paper600']:.3f} ece {r['ece600']:.3f} oof {r['oof_accuracy']:.3f}"
                     if "paper600" in r else r.get("error", "")))


def residual_analysis(items, cards, caches, paper600, test, recorded, pool) -> None:
    """What is left once topic_domain is in: errors by tier, label and the topic answer."""
    print("\n=== residuals ===")
    sports = ("practice", "team", "coach", "athlet", "game", "match", "training", "swim", "golf",
              "tennis", "row", "box", "player", "tournament", "season", "field", "gym", "skat",
              "baseball", "track", "soccer", "run")
    office = ("meeting", "office", "employee", "timesheet", "printer", "report", "deadline",
              "department", "manager", "conference", "email", "memo", "staff", "document",
              "schedul", "policy")

    def table(engine: str, config: str, feedback, ids, title: str):
        cache = caches[engine]
        fitted, info = one_fit(cards[config], cache, feedback)
        if fitted is None:
            print(title, info)
            return None
        score = fitted.score(SCORE)
        questions = fitted.questions()
        answers = cache.bulk_partial_answers(ids, questions)
        cells = collections.defaultdict(lambda: [0, 0])
        wrong = []
        for item_id in ids:
            item = items[item_id]
            result = predict(score, answers[item_id])
            topic = (answers[item_id].get("sentiment.topic_domain") or {}).get("choice", "-")
            hit = agrees(result.value, item.reference_label)
            key = (item.metadata["tier"], item.reference_label, topic)
            cells[key][0] += 1
            cells[key][1] += 0 if hit else 1
            if not hit:
                wrong.append((item, result, topic))
        print(f"\n{title}: {info.get('n_train')} labels, held-out acc "
              f"{1 - sum(c[1] for c in cells.values()) / len(ids):.3f} on {len(ids)}")
        print(f"  {'tier':8}{'label':10}{'topic':24}{'n':>6}{'errors':>8}{'err%':>7}")
        for key in sorted(cells):
            n, e = cells[key]
            if n >= 5:
                print(f"  {key[0]:8}{key[1]:10}{key[2]:24}{n:6d}{e:8d}{100 * e / n:6.0f}%")
        cue = collections.Counter()
        for item, result, topic in wrong:
            text = item.text.lower()
            cue[("sports-cue" if any(k in text for k in sports) else "no-sports-cue",
                 "office-cue" if any(k in text for k in office) else "no-office-cue",
                 item.reference_label)] += 1
        print("  errors by keyword cue x label:", dict(cue))
        print("  a few residual errors:")
        for item, result, topic in wrong[:12]:
            print(f"    [{item.metadata['tier']}/{item.reference_label}] said {result.value} "
                  f"({result.confidence:.2f}); topic={topic} :: {item.text[:110]}")
        return set(i.id for i, _, _ in wrong)

    table("jev", "H+T", recorded, paper600, "Jev H+T, the recorded 140 labels, paper600")
    big = simulated_feedback(items, pool)
    w_laya = table("laya", "H+T", big, test, "Laya H+T, all 5,280 pool labels, full test")
    w_laya7 = table("laya", "H+7+T", big, test, "Laya H+7+T, all 5,280 pool labels, full test")
    if w_laya is not None and w_laya7 is not None:
        print(f"\nLaya: H+T errors {len(w_laya)}, H+7+T errors {len(w_laya7)}, "
              f"fixed by the seven {len(w_laya - w_laya7)}, broken {len(w_laya7 - w_laya)}")
    w_hyb = table("jev+laya-topic", "H+T", big, test, "Jev holistic + Laya topic, all pool labels, full test")
    w_hyb7 = table("jev+laya-topic", "H+7+T", big, test, "Jev holistic + 7 + Laya topic, all pool labels, full test")
    if w_hyb is not None and w_hyb7 is not None:
        print(f"\nhybrid: H+T errors {len(w_hyb)}, H+7+T errors {len(w_hyb7)}, "
              f"fixed by the seven {len(w_hyb - w_hyb7)}, broken {len(w_hyb7 - w_hyb)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", type=Path, required=True,
                        help="where the two workspaces live (created by --prepare)")
    parser.add_argument("--prepare", action="store_true",
                        help="replay the recording and top up Laya's topic answers, then exit")
    parser.add_argument("--engines", nargs="+", default=["jev", "laya", "jev+laya-topic"])
    parser.add_argument("--out", type=Path, default=Path("studies/learning_curve.jsonl"))
    parser.add_argument("--no-residuals", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.scratch)
    else:
        main(args.scratch, args.out, args.engines, not args.no_residuals)
