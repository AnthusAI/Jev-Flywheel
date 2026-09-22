#!/usr/bin/env python
"""LF: gradient fine-tune Laya on the L1 seed-1 run's 140 labels, on the bios corpus.

    python scripts/finetune_laya_bios.py --seeds 1 2 3

``studies/PREREGISTERED.md``'s "does the engine read gender, and can the layer refuse to?"
section: "LF -- Laya fine-tuned on the same 140 labels, arm A's recipe from the [fine-tuning]
study above." Reuses the pure training/CV/calibration machinery from ``scripts/finetune_laya.py``
(arm A: full fine-tune, everything but ``act_head``/``temperature`` trainable) rather than
duplicating it, and points it at the bios corpus and the 140 labels
``fixtures/bios/recordings/L1-seed1`` recorded, instead of the sentiment corpus's own recording.

Recipe, unchanged from arm A: AdamW, weight decay 0.01, batch 16, 6% linear warmup then linear
decay, grad-clip 1.0, fp32, 10 epochs (n=140 <= 500). A learning rate is chosen once by 3-fold
CV (the same documented reduction from the pre-registered 5-fold that ``finetune_laya.py``
already uses) on seed 1 and reused for seeds 2 and 3, from {1e-5, 2e-5, 5e-5}. One temperature is
fit on the CV's out-of-fold logits.

Every run reloads the base checkpoint fresh (fine-tuning mutates weights in place) and bounds
MLX's buffer cache to 4 GB (``fresh_agent``), per this repo's GPU-hygiene rule.

Scored on the 2,000 held-out bios and their gender-swapped twins with
``scripts/bios_gender.py``, exactly like every other arm. Rows append to
``studies/bios_gender.jsonl`` (arm "LF"). Needs Apple silicon (``laya-mlx``); there is no way to
replay this arm without a GPU, unlike L1/L2, which is called out in the study's Outcome section.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, score_arm, write_rows  # noqa: E402
from finetune_laya import (  # noqa: E402
    DEFAULT_FOLDS, FULL_FT_LRS, cv_select_lr, epochs_for, fresh_agent, predict_logits,
    softmax, train_laya)

from jev_flywheel.items import FeedbackItem, JsonlStore, load_items, normalize_label  # noqa: E402
from jev_flywheel.laya import to_laya_question  # noqa: E402
from jev_flywheel.scorecard import Scorecard  # noqa: E402

FIXTURES = Path("fixtures/bios")
RECORDING = FIXTURES / "recordings" / "L1-seed1"
CLASSES = ("surgeon", "physician")           # index order for this corpus's binary question
FT_KIND = "full"                              # arm A's recipe: every parameter but act_head/temperature
OUT = Path("studies/bios_gender.jsonl")


def load_corpus():
    items = {i.id: i for i in load_items(FIXTURES / "items.jsonl")}
    card = Scorecard.from_yaml((FIXTURES / "scorecards" / "reference_full.yaml").read_text())
    question = to_laya_question(card.questions()["Occupation"])
    feedback = JsonlStore(RECORDING / "feedback.jsonl", FeedbackItem).all()
    assert len(feedback) == 140, f"expected 140 labels in {RECORDING}, found {len(feedback)}"
    train_ids = [f.item_id for f in feedback]
    train_labels = [CLASSES.index(normalize_label(f.final_answer_value)) for f in feedback]
    test_items = [i for i in items.values() if i.metadata["split"] == "test"]
    twin_items = [i for i in items.values() if i.metadata["split"] == "counterfactual"]
    return items, question, train_ids, train_labels, test_items, twin_items


def evaluate_held_out(agent, question, test_items, twin_items, temperature: float, seed: int,
                      n_labels: int):
    import numpy as np

    def verdicts_for(pool):
        logits = predict_logits(agent, [i.text for i in pool], question)
        cal = softmax(logits / temperature)
        pred_idx = cal.argmax(axis=1)
        out = {}
        for item, idx, probs in zip(pool, pred_idx, cal):
            predicted = CLASSES[idx]
            p_surgeon = float(probs[CLASSES.index("surgeon")])
            out[item.id] = Verdict(item.id, predicted, p_surgeon, item.reference_label,
                                   item.metadata.get("gender"))
        return out

    test_verdicts = verdicts_for(test_items)
    twin_verdicts = verdicts_for(twin_items)
    twins = {i.metadata["counterfactual_of"]: twin_verdicts[i.id] for i in twin_items}
    return score_arm(arm="LF", engine="laya", verdicts=list(test_verdicts.values()), twins=twins,
                     seed=seed, version=None, n_labels=n_labels, redacted=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    items, question, train_ids, train_labels, test_items, twin_items = load_corpus()
    texts = [items[i].text for i in train_ids]
    epochs = epochs_for(len(texts))

    print(f"CV (folds={args.folds}) on seed 1 to choose the learning rate "
         f"({len(texts)} labels, epochs={epochs})...")
    cv = cv_select_lr(FT_KIND, texts, train_labels, question, epochs=epochs,
                      candidates=FULL_FT_LRS, folds=args.folds, seed=1)
    print(f"  chosen lr={cv['chosen_lr']:g}, temperature={cv['temperature']:.3f}, "
         f"cv_mean={cv['cv_mean']}")

    for seed in args.seeds:
        print(f"seed {seed}: training (full fine-tune, {epochs} epochs, "
             f"lr={cv['chosen_lr']:g})...")
        agent = fresh_agent()
        train_laya(agent, FT_KIND, texts, train_labels, question, epochs=epochs,
                  lr=cv["chosen_lr"], seed=seed)
        metrics = evaluate_held_out(agent, question, test_items, twin_items, cv["temperature"],
                                    seed, len(texts))
        row = metrics.as_row()
        row["cv_folds"] = args.folds
        row["chosen_lr"] = cv["chosen_lr"]
        row["cv_mean"] = cv["cv_mean"]
        print(json.dumps(row, indent=2))
        write_rows([row], args.out)
        del agent


if __name__ == "__main__":
    main()
