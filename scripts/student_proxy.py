#!/usr/bin/env python
"""A cheap stand-in for the "distil the Jev head into a local text classifier" plan.

    python scripts/student_proxy.py --scratch /tmp/lc     # after scripts/learning_curve.py --prepare

No downloads, no model weights: the student is TF-IDF + logistic regression, which is the
weakest possible text-only learner, so its numbers are a floor for what a fine-tuned
encoder would do. What the script exercises is the *process*, not the model:

1. teacher  = the repo's own head, fit with ``fit_head`` on the 140 recorded human labels
              (Jev holistic + the seven cached elements + the topic element; the topic
              answers come from Laya for the pool, since Jev's exist for 740 items only)
2. teacher labels every pool item (hard label + probability)
3. two students: one trained on the teacher's labels, one on the corpus's reference labels
   (the latter is the "if we had 5,280 human labels" ceiling for this learner)
4. both are scored on the 3,521 held-out items against the reference label, against the
   teacher, per tier and per topic slice, and as a cascade: the student answers when its
   confidence clears a threshold and defers the rest.

Held-out items are never trained on and never label the teacher.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_flywheel.answers import AnswerCache  # noqa: E402
from jev_flywheel.cli import PACKAGED_FIXTURES  # noqa: E402
from jev_flywheel.evaluate import summarize  # noqa: E402
from jev_flywheel.fit import build_training_set, fit_head, with_fit  # noqa: E402
from jev_flywheel.items import FeedbackItem, JsonlStore, agrees  # noqa: E402
from jev_flywheel.scorecard import Scorecard  # noqa: E402
from jev_flywheel.scoring import predict  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from learning_curve import RECORDING, SCORE, make_card  # noqa: E402


def build_teacher(scratch: Path):
    """The teacher: the repo's head fitted on the 140 recorded labels, and what it says about every item.

    Returns ``(items, pool, test, t_pool, t_test, fit)``; ``t_*`` map item id to
    ``(label, confidence, topic)``.
    """
    jev = Workspace(scratch / "jev" / "var").require()
    laya = Workspace(scratch / "laya" / "var").require()
    items = {i.id: i for i in jev.items}
    pool = [i.id for i in jev.split("pool")]
    test = [i.id for i in jev.split("test")]
    v4 = jev.scorecard(4)
    topic_q = {k: v for k, v in v4.questions().items() if k != SCORE}
    hybrid = AnswerCache()
    for row in jev.cache.rows():
        hybrid.put_hashed(*row)
    for item_id, name, qhash, answer in laya.cache.rows():
        if name in topic_q:
            hybrid.put_hashed(item_id, name, qhash, answer)
    reference = Scorecard.from_yaml((PACKAGED_FIXTURES / "scorecards" / "reference_full.yaml").read_text())
    card = make_card(v4, reference, topic=True, seven=True)

    # 1. the teacher, from the 140 human labels the recording actually has
    recorded = JsonlStore(RECORDING / "feedback.jsonl", FeedbackItem).all()
    training = build_training_set(card.score(SCORE), card.questions(), hybrid, recorded)
    result = fit_head(training, card.score(SCORE))
    teacher = with_fit(card, SCORE, result)
    score = teacher.score(SCORE)

    def teach(ids):
        answers = hybrid.bulk_partial_answers(ids, teacher.questions())
        out = {}
        for i in ids:
            r = predict(score, answers[i])
            topic = (answers[i].get("sentiment.topic_domain") or {}).get("choice", "-")
            out[i] = (r.value, r.confidence, topic)
        return out

    t_pool, t_test = teach(pool), teach(test)
    acc = lambda pred, ids: sum(agrees(pred[i][0], items[i].reference_label) for i in ids) / len(ids)  # noqa: E731
    print(f"teacher: {result.n} labels, tier {result.tier.name}; accuracy vs reference "
          f"pool {acc(t_pool, pool):.3f}  test {acc(t_test, test):.3f}")
    return items, pool, test, t_pool, t_test, result


def main(scratch: Path) -> None:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    items, pool, test, t_pool, t_test, _ = build_teacher(scratch)

    # 2./3. students
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_pool = vec.fit_transform([items[i].text for i in pool])
    X_test = vec.transform([items[i].text for i in test])
    students = {}
    for name, labels in (("on-teacher-labels", [t_pool[i][0] for i in pool]),
                         ("on-reference-labels", [items[i].reference_label for i in pool])):
        clf = LogisticRegression(C=3.0, max_iter=3000).fit(X_pool, labels)
        proba = clf.predict_proba(X_test)
        pred = clf.classes_[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        students[name] = (pred, conf)

    # 4. scoring
    ref = np.array([items[i].reference_label for i in test])
    teach_pred = np.array([t_test[i][0] for i in test])
    tiers = np.array([items[i].metadata["tier"] for i in test])
    topics = np.array([t_test[i][2] for i in test])
    print(f"\n{'student':22}{'vs human':>10}{'vs teacher':>12}{'ECE(raw)':>10}")
    for name, (pred, conf) in students.items():
        s = summarize(conf.tolist(), (pred == ref).astype(int).tolist())
        print(f"{name:22}{(pred == ref).mean():10.3f}{(pred == teach_pred).mean():12.3f}{s.ece:10.3f}")
    print(f"{'teacher itself':22}{(teach_pred == ref).mean():10.3f}{1.0:12.3f}")

    print("\nper-slice accuracy vs human (student on teacher labels | teacher), n:")
    pred, conf = students["on-teacher-labels"]
    for key in sorted(set(zip(tiers, topics))):
        m = (tiers == key[0]) & (topics == key[1])
        if m.sum() >= 30:
            print(f"  {key[0]:8}{key[1]:24}{(pred[m] == ref[m]).mean():7.3f} | "
                  f"{(teach_pred[m] == ref[m]).mean():.3f}   n={m.sum()}")

    print("\ncascade (student on teacher labels answers above the threshold, teacher takes the rest):")
    print(f"  {'threshold':>9}{'coverage':>10}{'student acc':>13}{'cascade acc':>13}")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = conf >= thr
        cascade = np.where(m, pred, teach_pred)
        print(f"  {thr:9.2f}{m.mean():10.3f}{(pred[m] == ref[m]).mean() if m.any() else float('nan'):13.3f}"
              f"{(cascade == ref).mean():13.3f}")

    disagree = collections.Counter()
    for i, p in zip(test, pred):
        if p != t_test[i][0]:
            disagree[(items[i].metadata["tier"], t_test[i][2])] += 1
    print("\nstudent-teacher disagreements on held-out by (tier, topic):", dict(disagree.most_common(8)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", type=Path, required=True)
    main(parser.parse_args().scratch)
