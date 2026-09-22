#!/usr/bin/env python
"""Two exploratory extensions to studies/PREREGISTERED.md's fine-tune-Laya study. NOT
pre-registered; every row this writes carries ``"exploratory": true`` and is kept out of any
pre-registered tally.

(a) **Arm B at lower learning rates.** The pre-registered head-only grid {1e-4, 1e-3} looks too
    aggressive -- Arm B landed below untuned Laya (studies/finetune_laya.jsonl: 0.62-0.71 on
    paper-600 against raw Laya's 0.722). This tries {1e-5, 2e-5}, 3 seeds, on the recorded 140,
    reusing the pre-registered recipe (AdamW, wd 0.01, batch 16, warmup+linear decay, grad-clip
    1.0, fp32, 10 epochs at n=140) but skipping CV -- these two rates are picked to explore, not
    to be selected by cross-validation, so ``cv_folds`` is recorded as 0 and ``cv_note`` says so.
(b) **Arm C at very small n.** 20, 40, 80 random labels, 3 seeds, reusing lr=2e-05 (Arm C's own
    chosen rate at n=140) without its own CV, to find where full fine-tuning drops below the
    flywheel layer's 0.802 (paper-600) / 0.870 (full).

    python scripts/exploratory_finetune.py

Appends to ``studies/finetune_laya.jsonl`` (same file, same schema, distinguished only by
``exploratory: true``) so every reader of that file sees both together and has to look at the
flag rather than infer it from n or lr.

**Calibration caveat, exploratory rows only.** The pre-registered rows fit one temperature per
(arm, n) on the CV's out-of-fold logits. These rows skip CV entirely (the point is to try a
hand-picked rate, not to re-derive it), so there are no out-of-fold logits to fit a temperature
from; ``temperature`` is left at 1.0 (uncalibrated) and ``paper600_ece_calibrated`` /
``full_ece_calibrated`` on these rows equal the raw ECE, not a real calibration. Read
``paper600_accuracy`` / ``full_accuracy`` from these rows; do not compare their ECE columns
against the pre-registered rows' calibrated ECE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_laya as fl  # noqa: E402


def run(out: Path, folds: int) -> None:
    print("loading corpus (items, splits, recorded labels) ...")
    corpus = fl.Corpus()
    rows = []

    print("\n=== Exploratory (a): Arm B, lr in {1e-5, 2e-5}, seeds 1-3, n=140 ===")
    for lr in (1e-5, 2e-5):
        for seed in (1, 2, 3):
            print(f"\n--- B lr={lr:g} seed={seed} ---")
            # Bypass cv_select_lr: this lr is fixed by hand, not chosen by CV.
            fake_cv_cache = {("B", 140): {"chosen_lr": lr, "cv_scores": {}, "cv_mean": {},
                                          "temperature": 1.0, "folds": 0,
                                          "note": f"exploratory: lr={lr:g} fixed by hand, no CV"}}
            row = fl.run_laya_point(corpus, "B", 140, seed, "recorded", folds, fake_cv_cache)
            row["exploratory"] = True
            rows.append(row)
            fl.append_jsonl(out, [row])
            print(f"  paper600 acc {row['paper600_accuracy']:.3f}  full acc "
                  f"{row['full_accuracy']:.3f}  lr={row['chosen_lr']}  ({row['train_seconds']:.0f}s)")

    print("\n=== Exploratory (b): Arm C, n in {20, 40, 80}, seeds 1-3, lr=2e-05 (no CV) ===")
    for n in (20, 40, 80):
        for seed in (1, 2, 3):
            print(f"\n--- C n={n} seed={seed} ---")
            fake_cv_cache = {("C", n): {"chosen_lr": 2e-5, "cv_scores": {}, "cv_mean": {},
                                        "temperature": 1.0, "folds": 0,
                                        "note": "exploratory: lr=2e-05 reused from Arm C's "
                                                "n=140 choice, no CV at this size"}}
            row = fl.run_laya_point(corpus, "C", n, seed, "random-pool", folds, fake_cv_cache)
            row["exploratory"] = True
            rows.append(row)
            fl.append_jsonl(out, [row])
            print(f"  paper600 acc {row['paper600_accuracy']:.3f}  full acc "
                  f"{row['full_accuracy']:.3f}  lr={row['chosen_lr']}  ({row['train_seconds']:.0f}s)")

    print(f"\nwrote {len(rows)} exploratory rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("studies/finetune_laya.jsonl"))
    parser.add_argument("--folds", type=int, default=fl.DEFAULT_FOLDS,
                        help="unused (no CV runs here), kept for symmetry with finetune_laya.py")
    args = parser.parse_args()
    run(args.out, args.folds)
