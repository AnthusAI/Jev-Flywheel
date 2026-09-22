#!/usr/bin/env python
"""Exploratory: selective prediction across engines and layers. NOT pre-registered.

Coverage at >=95% accuracy (the largest fraction of items, ranked by confidence, that can be
auto-accepted while the accepted set stays >=95% correct) and AUROC of confidence vs. correct,
for: Jev alone, Jev + flywheel layer (v4), Laya alone, Laya + layer (v4), fine-tuned Laya (Arm
A, per seed), DistilBERT (Arm D). Reported with and without the neutral tier, on paper-600 and,
where per-item probabilities exist, the full 3,521 held-out items.

Jev alone/+layer and Laya alone/+layer are read straight from the recording (the same 140
labels replayed against both engines, as in ``scripts/laya_paired.py``) -- no GPU, no retraining,
just ``jev_flywheel.report``/``scoring`` machinery reused for its per-item predictions instead
of only the aggregate scoreboard. Jev's answers were only ever filled for paper-600 (asking Jev
about the full 3,521 costs money this repo does not spend), so Jev's full-test rows are omitted;
Laya is free, so both its rows are reported.

Arm A and Arm D rows need per-item held-out (item id, truth, calibrated P(positive)), which
``scripts/finetune_laya.py`` only writes when run with ``--save-probs``; look under
``var/finetune_laya/probs/`` and skip (with a note) whatever is missing rather than fabricate it.

Ties: Jev rounds its probabilities to two decimals, so many items land on the same confidence.
Coverage is computed over confidence-tied *blocks* (a tie is never split down the middle); the
number of items in the final, partially-admitted block is reported as ``tie_block_size``.

    python scripts/selective_prediction.py

Writes ``studies/selective_prediction.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.items import agrees
from jev_flywheel.recording import replay
from jev_flywheel.report import complete_items

RECORDING = Path(PACKAGED_FIXTURES) / "recordings" / "simulated-labeler"
PROBS_DIR = Path("var/finetune_laya/probs")
MIN_ACCURACY = 0.95


# ---- per-item records, from a replayed recording (Jev / Laya, any version) -------------------

def per_item_records(workspace, score_name: str, version: int, split: str,
                     item_ids: Optional[set] = None) -> List[dict]:
    from jev_flywheel.scoring import predict

    card = workspace.scorecard(version)
    score = card.score(score_name)
    wanted = set(card.questions())
    items = [i for i in workspace.split(split) if i.reference_label is not None
             and (item_ids is None or i.id in item_ids)]
    answers = workspace.cache.bulk_partial_answers([i.id for i in items], card.questions())
    out = []
    for item in items:
        ans = answers[item.id]
        if not (wanted <= set(ans)):
            continue          # incomplete answers for this version: excluded, not imputed
        result = predict(score, ans)
        hit = int(agrees(result.value, item.reference_label))
        out.append({"item_id": item.id, "confidence": float(result.confidence or 0.0),
                    "correct": hit, "tier": str(item.metadata.get("tier", "?"))})
    return out


# ---- per-item records, from a finetune_laya.py --save-probs dump -----------------------------

def per_item_records_from_probs(probs_path: Path, workspace_items: Dict[str, object]) -> List[dict]:
    out = []
    for line in probs_path.read_text().splitlines():
        row = json.loads(line)
        item = workspace_items.get(row["item_id"])
        tier = str(item.metadata.get("tier", "?")) if item is not None else "?"
        p_positive = row["p_positive"]
        pred = "positive" if p_positive >= 0.5 else "negative"
        confidence = max(p_positive, 1.0 - p_positive)
        hit = int(pred == row["truth"])
        out.append({"item_id": row["item_id"], "confidence": confidence, "correct": hit,
                    "tier": tier})
    return out


# ---- selective-prediction metrics, pure Python -------------------------------------------------

def coverage_at_min_accuracy(records: Sequence[dict], min_acc: float = MIN_ACCURACY) -> dict:
    """Largest fraction of ``records`` (ranked by confidence, tied blocks kept whole) that can be
    accepted while the accepted set's accuracy stays >= ``min_acc``."""
    if not records:
        return {"coverage": None, "n_accepted": 0, "accepted_accuracy": None, "tie_block_size": 0}
    order = sorted(records, key=lambda r: -r["confidence"])
    blocks: List[List[dict]] = []
    for r in order:
        if blocks and blocks[-1][0]["confidence"] == r["confidence"]:
            blocks[-1].append(r)
        else:
            blocks.append([r])
    n_correct = n_total = 0
    best_coverage, best_acc, best_tie_block = 0.0, None, 0
    for block in blocks:
        new_total = n_total + len(block)
        new_correct = n_correct + sum(b["correct"] for b in block)
        if new_correct / new_total >= min_acc:
            n_total, n_correct = new_total, new_correct
            best_coverage = n_total / len(records)
            best_acc = n_correct / n_total
            best_tie_block = len(block)
        else:
            break
    return {"coverage": round(best_coverage, 4), "n_accepted": n_total,
            "accepted_accuracy": round(best_acc, 4) if best_acc is not None else None,
            "tie_block_size": best_tie_block}


def auroc(records: Sequence[dict]) -> Optional[float]:
    """AUROC of confidence vs. correctness (Mann-Whitney U / rank-sum, average ranks for ties).
    None when every item is correct or every item is wrong (the statistic is undefined)."""
    n_pos = sum(r["correct"] for r in records)
    n_neg = len(records) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ordered = sorted(records, key=lambda r: r["confidence"])
    n = len(ordered)
    rank_of = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and ordered[j]["confidence"] == ordered[i]["confidence"]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            rank_of[k] = avg_rank
        i = j
    rank_sum_pos = sum(rank_of[k] for k in range(n) if ordered[k]["correct"])
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return round(u / (n_pos * n_neg), 4)


def summarize_records(records: Sequence[dict]) -> dict:
    if not records:
        return {"n": 0, "accuracy": None, "coverage_at_95": None, "auroc": None}
    accuracy = sum(r["correct"] for r in records) / len(records)
    return {
        "n": len(records), "accuracy": round(accuracy, 4),
        **{f"coverage_at_95_{k}": v for k, v in coverage_at_min_accuracy(records).items()},
        "auroc": auroc(records),
    }


def rows_for(model: str, seed: Optional[int], sample: str, records: Sequence[dict]) -> List[dict]:
    with_neutral = summarize_records(records)
    without_neutral = summarize_records([r for r in records if r["tier"] != "neutral"])
    return [{"exploratory": True, "model": model, "seed": seed, "sample": sample,
             "neutral_tier": "included", **with_neutral},
            {"exploratory": True, "model": model, "seed": seed, "sample": sample,
             "neutral_tier": "excluded", **without_neutral}]


def main(out: Path) -> None:
    rows: List[dict] = []

    # --- Jev and Laya, alone (v1, no decision head) and with the layer (v4, steered) ----------
    # The same fixed 600-item sample every headline number in this repo uses (README, 'paper-600'),
    # from the *latest* version's completion (v4, the strictest: it also needs topic_domain) so
    # v1 and v4 are compared on identical items, as scripts/laya_paired.py does.
    jev = replay(RECORDING, Path(tempfile.mkdtemp(prefix="selpred-jev-")) / "var",
                Path(PACKAGED_FIXTURES))
    paper600_ids = complete_items(jev, "test")
    assert len(paper600_ids) == 600, len(paper600_ids)
    score = jev.scorecard().scores[0].name
    rows += rows_for("jev_alone", None, "paper-600",
                     per_item_records(jev, score, 1, "test", paper600_ids))
    rows += rows_for("jev_plus_layer", None, "paper-600",
                     per_item_records(jev, score, 4, "test", paper600_ids))
    print(f"jev: {len(paper600_ids)} paper-600 items scored (v1 and v4); "
          f"full-test skipped, Jev was never asked about all 3,521")

    import asyncio

    from jev_flywheel.jev import JevSession
    from jev_flywheel.laya import LayaClient

    client = LayaClient()
    client.warm()
    laya = replay(RECORDING, Path(tempfile.mkdtemp(prefix="selpred-laya-")) / "var",
                 Path(PACKAGED_FIXTURES), answers="answers-laya.jsonl.gz", engine="laya",
                 client_factory=lambda: client, on_step=lambda m: None)
    # v4 needs topic_domain, which answers-laya.jsonl.gz only ships for items already asked
    # during the recorded run; top it up on the rest (as scripts/laya_paired.py does) so v4 can
    # be scored on the same paper-600/full-3521 item sets as v1. Laya inference, not training --
    # still real GPU/Metal work, so this is skipped while a finetune_laya.py job is running.
    laya_full_ids = {i.id for i in laya.split("test")}
    questions = laya.scorecard().questions()
    report = asyncio.run(laya.cache.fill(JevSession(client_factory=lambda: client),
                                         laya.split("test"), questions))
    print(f"laya top-up over {len(laya_full_ids)} test items: {report.requested} asked, "
          f"{report.failures} failed")
    for name, version in (("laya_alone", 1), ("laya_plus_layer", 4)):
        rows += rows_for(name, None, "paper-600",
                         per_item_records(laya, score, version, "test", paper600_ids))
        rows += rows_for(name, None, "full-3521",
                         per_item_records(laya, score, version, "test", laya_full_ids))
    print(f"laya: {len(paper600_ids)} paper-600 items, {len(laya_full_ids)} full-test items "
          "(Laya is free, so both samples are the whole split)")

    # --- Arm A / Arm D, from finetune_laya.py --save-probs dumps -------------------------------
    workspace_items = {i.id: i for i in jev.items}
    if not PROBS_DIR.exists():
        print(f"no per-item probability dumps under {PROBS_DIR} -- run "
              "scripts/finetune_laya.py ... --save-probs first; Arm A/D rows skipped")
    else:
        for arm, label in (("A", "finetuned_laya_arm_A"), ("D", "distilbert_arm_D")):
            for seed in (1, 2, 3):
                for split, sample in (("paper600", "paper-600"), ("full", "full-3521")):
                    path = PROBS_DIR / f"{arm}-seed{seed}-n140-{split}.jsonl"
                    if not path.exists():
                        print(f"missing {path}, skipped")
                        continue
                    records = per_item_records_from_probs(path, workspace_items)
                    rows += rows_for(label, seed, sample, records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:      # a study is one run: rewrite, not append
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} rows to {out}")
    for r in rows:
        if r["n"]:
            print(f"{r['model']:22} seed={str(r['seed']):4} {r['sample']:10} "
                  f"{r['neutral_tier']:9} n={r['n']:5} acc={r['accuracy']:.3f} "
                  f"cov@95={r['coverage_at_95_coverage']} auroc={r['auroc']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("studies/selective_prediction.jsonl"))
    main(parser.parse_args().out)
