#!/usr/bin/env python
"""Does ordinary gradient fine-tuning of Laya beat the fitted-head layer, on the same labels?

    python scripts/finetune_laya.py --arms A B D --seeds 1 2 3            # M2, n=140
    python scripts/finetune_laya.py --arms C --seeds 1 2 --sizes 140 300 500 800 2000 5140  # M3
    python scripts/finetune_laya.py --arms A --seeds 1 --drift            # M4, after an A run

Pre-registered in ``studies/PREREGISTERED.md`` ("fine-tuning Laya on the same labels"), written
before any arm here was trained or evaluated. The README states plainly that the flywheel was
never compared against actual gradient fine-tuning on the same label budget; this script is
that comparison, on Laya (``laya-mlx``, a 421M-parameter ModernBERT-large encoder) because it
is the one engine in this repo cheap enough to fine-tune on a laptop.

Process, end to end:

1. **Data.** The corpus's own reference label is the simulated labeler everywhere in this repo
   (``scripts/learning_curve.py``), so it is the target here too. Arms A and B use the
   recording's 140 *actively selected* human labels (``fixtures/recordings/simulated-labeler``,
   the same 140 the README's headline numbers were fit on). Arm C draws uniformly at random
   from the pool *minus* those 140 items (5,140 of the pool's 5,280 items), so a draw of
   n=5,140 is the entire remaining pool and the largest arm-C point is not really "random" —
   it is everything. Held-out sets (``paper600``, the README's 600 items, and ``full``, all
   3,521 test items) are never labeled, never selected from, and never seen by any lr or epoch
   choice.
2. **Recipe, fixed in the pre-registration.** AdamW, weight decay 0.01, batch 16, 6% linear
   warmup then linear decay to 0, grad-clip 1.0, fp32. Epochs 10 for n<=500, 3 above. A
   learning rate is chosen once per (arm, n) by k-fold cross-validation *inside the training
   labels only*, on seed 1, and reused for the other seeds. The pre-registration specified
   5-fold CV; this script uses **3-fold** throughout to keep wall-clock affordable on one M1 Max
   GPU (documented reduction, not a silent one -- see the ``--folds`` default and the README of
   whichever study writes this up). The pre-registration lets n=5,140 reuse the rate chosen at
   n=2,000. ``--cv-anchors`` cuts further: only the listed Arm C sizes run CV and every other
   size reuses the nearest smaller anchor's rate *and temperature*. The recorded run used
   ``--cv-anchors 140 800`` (one anchor per epoch regime), which is a deviation from the
   pre-registration; each row's ``cv_note`` says where its rate came from.
3. **Calibration.** One temperature, fit by ``fit_temperature`` (borrowed from
   ``scripts/distill_student.py``) on the out-of-fold logits from the *same* CV that chose the
   learning rate -- no extra runs, no held-out item ever touches this either.
4. **Freezing.** Arm A/C (full fine-tune) trains everything except ``act_head`` (never used by
   this loss) and ``temperature`` (a shipped calibration buffer, not a parameter this experiment
   means to fit by gradient). Arm B (head-only) trains ``DecisionHead``, ``type_emb`` and
   ``scorer``; the ModernBERT encoder is frozen. Arm D is ``distilbert-base-uncased``, fine-tuned
   with ``train_student`` from ``scripts/distill_student.py`` on hard labels, as the "ordinary
   fine-tune, ordinary small model" baseline.
5. **A fresh checkpoint every run.** Fine-tuning mutates weights in place, so every seed and
   every fold reloads ``jev_flywheel.laya.DEFAULT_CHECKPOINT`` from scratch. Nothing here ever
   saves a Laya checkpoint to the repo; ``--scratch`` (default ``var/finetune_laya``) is
   ``.gitignore``d, and a 1.7 GB checkpoint is not written there either unless ``--save-checkpoints``
   is passed, for spot-checking only.

Rows go to ``studies/finetune_laya.jsonl`` (M2/M3) and ``studies/finetune_laya_drift.jsonl``
(M4, ``--drift``), one row per (arm, seed, n) or (arm, seed, question). Pass ``--save-probs`` to
also dump per-item held-out (item id, truth, calibrated P(positive)) under
``--scratch/probs/<arm>-seed<seed>-n<n>-<split>.jsonl`` -- needed by the separate
selective-prediction comparison (``scripts/selective_prediction.py``), never written under
``studies/`` and never committed.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_flywheel.cli import PACKAGED_FIXTURES  # noqa: E402
from jev_flywheel.evaluate import summarize  # noqa: E402
from jev_flywheel.items import FeedbackItem, Item, JsonlStore, normalize_label  # noqa: E402
from jev_flywheel.laya import DEFAULT_CHECKPOINT, to_laya_question  # noqa: E402
from jev_flywheel.recording import replay  # noqa: E402
from jev_flywheel.report import complete_items  # noqa: E402
from jev_flywheel.scorecard import Scorecard  # noqa: E402

RECORDING = Path(PACKAGED_FIXTURES) / "recordings" / "simulated-labeler"
CLASSES = ("positive", "negative")            # index order matches the Sentiment question
FULL_FT_LRS = (1e-5, 2e-5, 5e-5)
HEAD_FT_LRS = (1e-4, 1e-3)
STUDENT_LRS = (2e-5, 5e-5)
BATCH = 16
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_FRACTION = 0.06
DEFAULT_FOLDS = 3            # reduced from the pre-registered 5, for wall-clock; see the docstring
# Set from main() when --save-checkpoints is passed. Never the repo: a checkpoint is ~1.7 GB.
CHECKPOINT_DIR: Optional[Path] = None
# Set from main() when --save-probs is passed. Per-item held-out probabilities for the
# selective-prediction comparison (M6); under var/, never studies/, never committed.
PROBS_DIR: Optional[Path] = None


# ---- data -----------------------------------------------------------------------------------

class Corpus:
    """Items, splits and the recorded human labels, loaded once and reused everywhere."""

    def __init__(self):
        jev = replay(RECORDING, Path(tempfile.mkdtemp(prefix="finetune-laya-")) / "var",
                     Path(PACKAGED_FIXTURES))
        self.items: Dict[str, Item] = {i.id: i for i in jev.items}
        self.pool: List[str] = [i.id for i in jev.split("pool")]
        self.test: List[str] = [i.id for i in jev.split("test")]
        self.paper600: List[str] = sorted(complete_items(jev, "test"))
        assert len(self.paper600) == 600, len(self.paper600)
        recorded = JsonlStore(RECORDING / "feedback.jsonl", FeedbackItem).all()
        self.recorded_ids: List[str] = [f.item_id for f in recorded]
        self.recorded_labels: Dict[str, str] = {
            f.item_id: normalize_label(f.final_answer_value) for f in recorded}
        assert len(self.recorded_ids) == 140, len(self.recorded_ids)
        # Arm C's universe: the pool minus the 140 items Arms A/B already spent (5,140 items).
        recorded_set = set(self.recorded_ids)
        self.random_universe: List[str] = [i for i in self.pool if i not in recorded_set]
        assert len(self.random_universe) == 5140, len(self.random_universe)
        card = Scorecard.from_yaml((Path(PACKAGED_FIXTURES) / "scorecards" /
                                    "reference_full.yaml").read_text())
        self.sentiment_question = to_laya_question(card.questions()["Sentiment"])
        self.reference_questions = {
            k: to_laya_question(v) for k, v in card.questions().items() if k != "Sentiment"}
        # The element the recorded steering round discovered (the recording's v4 scorecard).
        self.topic_question = {
            k: to_laya_question(v) for k, v in jev.scorecard(4).questions().items()
            if k.endswith("topic_domain")}

    def label_index(self, item_id: str, source: str) -> int:
        label = self.recorded_labels[item_id] if source == "recorded" else \
            self.items[item_id].reference_label
        return CLASSES.index(label)

    def recorded_prefix_or_random(self, n: int, source: str, seed: int) -> List[str]:
        if source == "recorded":
            assert n == len(self.recorded_ids), "recorded labels are used all at once"
            return list(self.recorded_ids)
        universe = self.random_universe
        if n >= len(universe):
            return list(universe)
        return random.Random(10_000 * seed + n).sample(universe, n)


# ---- Laya batches and training ---------------------------------------------------------------

def fresh_agent():
    """Base weights, reloaded. Also bounds MLX's buffer cache: left alone it grew to ~24 GB over
    a training run on a 32 GB machine, pushed the system into swap, and made a fold take five
    times as long as it should."""
    import gc

    import laya_mlx
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()
    mx.set_cache_limit(4 * 1024 ** 3)
    return laya_mlx.load(DEFAULT_CHECKPOINT, dtype="float32", compile=False)


def prepare_rows(agent, texts: Sequence[str], question: dict) -> List[dict]:
    """``question`` is the wire-format Sentiment question (``to_laya_question`` output)."""
    from laya_mlx.common import build_sequence
    q = agent._to_internal(question)
    out = []
    for text in texts:
        ids, markers = build_sequence(agent.tok, text, q, agent.cfg.get("max_len", 512),
                                       agent.cfg.get("head_max_len", 192))
        out.append({"ids": ids, "markers": markers, "qtype": 0})
    return out


def set_trainable(model, arm: str) -> None:
    """Arm 'full': everything but act_head/temperature. Arm 'head': only the decision head."""
    if arm == "full":
        model.unfreeze(recurse=True)
        model.act_head.freeze(recurse=True)
        model.freeze(keys=["temperature"], recurse=False)
    elif arm == "head":
        model.freeze(recurse=True)
        model.head.unfreeze(recurse=True)
        model.type_emb.unfreeze(recurse=True)
        model.scorer.unfreeze(recurse=True)
    else:
        raise ValueError(arm)


def batches(n: int, batch_size: int, rng: random.Random):
    order = list(range(n))
    rng.shuffle(order)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def train_laya(agent, arm: str, texts: List[str], labels: List[int], question: dict, *,
               epochs: int, lr: float, seed: int) -> float:
    """Fine-tune ``agent.model`` in place. Returns wall-clock training seconds."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from laya_mlx.agent import collate_items

    set_trainable(agent.model, arm)
    rows = prepare_rows(agent, texts, question)
    n = len(rows)
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(WARMUP_FRACTION * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    def loss_fn(model, batch, target):
        logits, _ = model(**batch)
        return nn.losses.cross_entropy(logits[:, :2], target, reduction="mean")

    loss_and_grad = nn.value_and_grad(agent.model, loss_fn)
    optimizer = optim.AdamW(learning_rate=lr_at(0), weight_decay=WEIGHT_DECAY)
    rng = random.Random(seed)
    started = time.time()
    step = 0
    for _epoch in range(epochs):
        for idx in batches(n, BATCH, rng):
            chunk = [rows[i] for i in idx]
            target = mx.array([labels[i] for i in idx])
            batch = collate_items(chunk, agent.tok.pad_token_id,
                                   max_length=agent.cfg.get("max_len", 512))
            tensors = {k: mx.array(v) for k, v in batch.items()}
            loss, grads = loss_and_grad(agent.model, tensors, target)
            grads = clip_grad_norm(grads, GRAD_CLIP)
            optimizer.learning_rate = lr_at(step)
            optimizer.update(agent.model, grads)
            mx.eval(agent.model.parameters(), loss)
            step += 1
    return time.time() - started


def clip_grad_norm(grads, max_norm: float):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    flat = [v for _, v in tree_flatten(grads) if v is not None]
    if not flat:
        return grads
    total = mx.sqrt(sum((g.astype(mx.float32) ** 2).sum() for g in flat))
    scale = mx.minimum(mx.array(1.0), max_norm / mx.maximum(total, 1e-6))
    return tree_map(lambda g: g * scale if g is not None else g, grads)


def predict_logits(agent, texts: Sequence[str], question: dict, batch_size: int = 32):
    """Raw (pre-temperature) two-class logits for ``texts`` under the Sentiment question."""
    import mlx.core as mx
    import numpy as np
    from laya_mlx.agent import collate_items

    rows = prepare_rows(agent, texts, question)
    out = np.zeros((len(rows), 2), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = collate_items(chunk, agent.tok.pad_token_id, max_length=agent.cfg.get("max_len", 512))
        tensors = {k: mx.array(v) for k, v in batch.items()}
        logits, _ = agent.model(**tensors)
        mx.eval(logits)
        out[start:start + len(chunk)] = np.array(logits[:, :2])
    return out


# ---- calibration + metrics --------------------------------------------------------------------

def fit_temperature(logits, truth):
    import numpy as np
    from scipy.optimize import minimize_scalar

    def nll(log_t):
        z = logits / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(truth)), truth].mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def softmax(z):
    import numpy as np
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calibrated_positive_probabilities(logits, temperature: float):
    """Per-item calibrated P(positive) (``CLASSES[0]``), for the held-out-probability dump used
    by the selective-prediction comparison. Pure numpy; no model, no I/O."""
    cal_p = softmax(logits / temperature)
    return cal_p[:, CLASSES.index("positive")]


def save_item_probabilities(path: Path, ids: Sequence[str], truth_idx: Sequence[int],
                            logits, temperature: float) -> None:
    """Item id, truth label, and calibrated P(positive) for one (arm, seed, n, split) point, one
    JSON object per line. Goes under ``var/`` (``--scratch``, .gitignore'd) -- never under
    ``studies/`` and never committed. Needed by the selective-prediction comparison (M6), which
    ranks items by confidence and needs the per-item probability, not just split-level summaries."""
    import numpy as np

    p_positive = calibrated_positive_probabilities(np.asarray(logits), temperature)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item_id, truth, p in zip(ids, truth_idx, p_positive):
            handle.write(json.dumps({"item_id": item_id, "truth": CLASSES[truth],
                                     "p_positive": round(float(p), 6)}) + "\n")


def score_split(logits, truth_idx, tiers, temperature: float) -> dict:
    import numpy as np
    raw_p = softmax(logits)
    cal_p = softmax(logits / temperature)
    pred = raw_p.argmax(axis=1)
    correct = (pred == np.array(truth_idx)).astype(int).tolist()
    raw_conf = raw_p.max(axis=1).tolist()
    cal_conf = cal_p.max(axis=1).tolist()
    s_raw = summarize(raw_conf, correct)
    s_cal = summarize(cal_conf, correct)
    by_tier = collections.defaultdict(list)
    for c, t in zip(correct, tiers):
        by_tier[t].append(c)
    return {
        "accuracy": round(s_raw.accuracy, 4),
        "ece_raw": round(s_raw.ece, 4),
        "ece_calibrated": round(s_cal.ece, 4),
        "brier_calibrated": round(s_cal.brier, 4),
        "temperature": round(temperature, 4),
        "by_tier": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_tier.items())},
    }


# ---- k-fold CV for the learning rate -----------------------------------------------------------

def kfold_indices(n: int, folds: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    out = []
    for f in range(folds):
        val = order[f::folds]
        val_set = set(val)
        train = [i for i in order if i not in val_set]
        out.append((train, val))
    return out


def cv_select_lr(arm: str, texts: List[str], labels: List[int], question: dict, *, epochs: int,
                  candidates: Sequence[float], folds: int, seed: int) -> dict:
    """K-fold CV inside the training labels only. Returns the chosen lr, per-lr mean accuracy,
    and the winning lr's out-of-fold logits/labels for temperature fitting."""
    import numpy as np

    n = len(texts)
    folds_idx = kfold_indices(n, folds, seed)
    scores: Dict[float, List[float]] = {}
    oof_by_lr: Dict[float, np.ndarray] = {}
    for lr in candidates:
        oof_logits = np.zeros((n, 2), dtype=np.float32)
        accs = []
        for train_idx, val_idx in folds_idx:
            agent = fresh_agent()
            train_laya(agent, arm, [texts[i] for i in train_idx], [labels[i] for i in train_idx],
                       question, epochs=epochs, lr=lr, seed=seed)
            logits = predict_logits(agent, [texts[i] for i in val_idx], question)
            pred = logits.argmax(axis=1)
            truth = np.array([labels[i] for i in val_idx])
            accs.append(float((pred == truth).mean()))
            print(f"    cv lr={lr:g} fold {len(accs)}/{folds}: acc {accs[-1]:.3f}", flush=True)
            for row, i in enumerate(val_idx):
                oof_logits[i] = logits[row]
            del agent
        scores[lr] = accs
        oof_by_lr[lr] = oof_logits
    means = {lr: sum(v) / len(v) for lr, v in scores.items()}
    best_lr = max(means, key=means.get)
    temperature = fit_temperature(oof_by_lr[best_lr], np.array(labels))
    return {
        "chosen_lr": best_lr,
        "cv_scores": {str(lr): [round(a, 4) for a in accs] for lr, accs in scores.items()},
        "cv_mean": {str(lr): round(m, 4) for lr, m in means.items()},
        "temperature": temperature,
        "folds": folds,
    }


def epochs_for(n: int) -> int:
    return 10 if n <= 500 else 3


ARM_KIND = {"A": "full", "B": "head", "C": "full"}
ARM_LRS = {"A": FULL_FT_LRS, "B": HEAD_FT_LRS, "C": FULL_FT_LRS}


# ---- one Laya arm, one (n, seed) point ---------------------------------------------------------

def run_laya_point(corpus: Corpus, arm: str, n: int, seed: int, label_source: str, folds: int,
                    cv_cache: Dict[Tuple[str, int], dict], reuse_lr_from: Optional[int] = None
                    ) -> dict:
    ids = corpus.recorded_prefix_or_random(n, label_source, seed)
    texts = [corpus.items[i].text for i in ids]
    labels = [corpus.label_index(i, label_source) for i in ids]
    epochs = epochs_for(n)
    kind = ARM_KIND[arm]
    question = corpus.sentiment_question

    cache_key = (arm, n)
    if cache_key not in cv_cache:
        if reuse_lr_from is not None and (arm, reuse_lr_from) in cv_cache:
            reused = cv_cache[(arm, reuse_lr_from)]
            print(f"  [{arm} n={n}] reusing the lr ({reused['chosen_lr']:g}) and temperature chosen at "
                  f"n={reuse_lr_from}; no CV at this size (a wall-clock cut: the pre-registration "
                  "allows it only for n=5,140 from n=2,000)")
            cv_cache[cache_key] = dict(reused, folds=0,
                                       note=f"lr and temperature reused from n={reuse_lr_from}")
        else:
            print(f"  [{arm} n={n}] {folds}-fold CV over {ARM_LRS[arm]} (seed 1) ...")
            cv_ids = ids if seed == 1 else corpus.recorded_prefix_or_random(n, label_source, 1)
            cv_texts = [corpus.items[i].text for i in cv_ids]
            cv_labels = [corpus.label_index(i, label_source) for i in cv_ids]
            cv_cache[cache_key] = cv_select_lr(kind, cv_texts, cv_labels, question, epochs=epochs,
                                               candidates=ARM_LRS[arm], folds=folds, seed=1)
    cv = cv_cache[cache_key]
    lr = cv["chosen_lr"]

    agent = fresh_agent()
    train_seconds = train_laya(agent, kind, texts, labels, question, epochs=epochs, lr=lr, seed=seed)
    if CHECKPOINT_DIR is not None:
        ckpt = CHECKPOINT_DIR / f"{arm}-seed{seed}-n{n}.safetensors"
        agent.model.save_weights(str(ckpt))
        print(f"  saved checkpoint (spot-check only, not committed): {ckpt}")
    p600_logits = predict_logits(agent, [corpus.items[i].text for i in corpus.paper600], question)
    full_logits = predict_logits(agent, [corpus.items[i].text for i in corpus.test], question)
    del agent

    p600_truth = [corpus.label_index(i, "reference") for i in corpus.paper600]
    full_truth = [corpus.label_index(i, "reference") for i in corpus.test]
    p600_tiers = [corpus.items[i].metadata.get("tier") for i in corpus.paper600]
    full_tiers = [corpus.items[i].metadata.get("tier") for i in corpus.test]
    temperature = cv["temperature"]

    row = {
        "arm": arm, "seed": seed, "n_labels": n, "label_source": label_source,
        "chosen_lr": lr, "cv_scores": cv["cv_scores"], "cv_mean": cv["cv_mean"],
        "cv_folds": cv["folds"], "cv_note": cv.get("note", "own CV"), "epochs": epochs,
        "train_seconds": round(train_seconds, 1),
    }
    p600 = score_split(p600_logits, p600_truth, p600_tiers, temperature)
    full = score_split(full_logits, full_truth, full_tiers, temperature)
    row.update({f"paper600_{k}": v for k, v in p600.items()})
    row.update({f"full_{k}": v for k, v in full.items()})
    if PROBS_DIR is not None:
        save_item_probabilities(PROBS_DIR / f"{arm}-seed{seed}-n{n}-paper600.jsonl",
                                corpus.paper600, p600_truth, p600_logits, temperature)
        save_item_probabilities(PROBS_DIR / f"{arm}-seed{seed}-n{n}-full.jsonl",
                                corpus.test, full_truth, full_logits, temperature)
    return row


# ---- Arm D: DistilBERT on the same 140, via the repo's own train_student ----------------------

def run_student_point(corpus: Corpus, seed: int, folds: int,
                       cv_cache: Dict[Tuple[str, int], dict]) -> dict:
    import numpy as np
    from distill_student import fit_temperature as student_fit_temperature
    from distill_student import logits_for, train_student

    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    ids = corpus.recorded_ids
    texts = [corpus.items[i].text for i in ids]
    labels = [corpus.label_index(i, "recorded") for i in ids]
    targets = np.array([[1.0, 0.0] if lab == 0 else [0.0, 1.0] for lab in labels])
    model_name = "distilbert-base-uncased"
    max_len = 96
    epochs = epochs_for(len(ids))

    cache_key = ("D", len(ids))
    if cache_key not in cv_cache:
        print(f"  [D n={len(ids)}] {folds}-fold CV over {STUDENT_LRS} (seed 1) ...")
        folds_idx = kfold_indices(len(ids), folds, 1)
        scores, oof_by_lr = {}, {}
        for lr in STUDENT_LRS:
            oof = np.zeros((len(ids), 2), dtype=np.float32)
            accs = []
            for train_idx, val_idx in folds_idx:
                tok, model = train_student(model_name, [texts[i] for i in train_idx],
                                           targets[train_idx], epochs=epochs, lr=lr, batch=16,
                                           seed=1, device=device, max_len=max_len)
                lg = logits_for(tok, model, [texts[i] for i in val_idx], device=device, max_len=max_len)
                pred = lg.argmax(axis=1)
                truth = np.array([labels[i] for i in val_idx])
                accs.append(float((pred == truth).mean()))
                for row, i in enumerate(val_idx):
                    oof[i] = lg[row]
                del model
            scores[lr] = accs
            oof_by_lr[lr] = oof
        means = {lr: sum(v) / len(v) for lr, v in scores.items()}
        best_lr = max(means, key=means.get)
        temperature = student_fit_temperature(oof_by_lr[best_lr], np.array(labels))
        cv_cache[cache_key] = {
            "chosen_lr": best_lr,
            "cv_scores": {str(lr): [round(a, 4) for a in accs] for lr, accs in scores.items()},
            "cv_mean": {str(lr): round(m, 4) for lr, m in means.items()},
            "temperature": temperature, "folds": folds,
        }
    cv = cv_cache[cache_key]
    lr = cv["chosen_lr"]

    started = time.time()
    tok, model = train_student(model_name, texts, targets, epochs=epochs, lr=lr, batch=16,
                               seed=seed, device=device, max_len=max_len)
    train_seconds = time.time() - started
    p600_logits = logits_for(tok, model, [corpus.items[i].text for i in corpus.paper600],
                             device=device, max_len=max_len)
    full_logits = logits_for(tok, model, [corpus.items[i].text for i in corpus.test],
                             device=device, max_len=max_len)
    del model

    p600_truth = [corpus.label_index(i, "reference") for i in corpus.paper600]
    full_truth = [corpus.label_index(i, "reference") for i in corpus.test]
    p600_tiers = [corpus.items[i].metadata.get("tier") for i in corpus.paper600]
    full_tiers = [corpus.items[i].metadata.get("tier") for i in corpus.test]
    temperature = cv["temperature"]

    row = {
        "arm": "D", "seed": seed, "n_labels": len(ids), "label_source": "recorded",
        "chosen_lr": lr, "cv_scores": cv["cv_scores"], "cv_mean": cv["cv_mean"],
        "cv_folds": cv["folds"], "epochs": epochs, "train_seconds": round(train_seconds, 1),
        "model": model_name,
    }
    p600 = score_split(p600_logits, p600_truth, p600_tiers, temperature)
    full = score_split(full_logits, full_truth, full_tiers, temperature)
    row.update({f"paper600_{k}": v for k, v in p600.items()})
    row.update({f"full_{k}": v for k, v in full.items()})
    if PROBS_DIR is not None:
        save_item_probabilities(PROBS_DIR / f"D-seed{seed}-n{len(ids)}-paper600.jsonl",
                                corpus.paper600, p600_truth, p600_logits, temperature)
        save_item_probabilities(PROBS_DIR / f"D-seed{seed}-n{len(ids)}-full.jsonl",
                                corpus.test, full_truth, full_logits, temperature)
    return row


# ---- M4: drift probe -----------------------------------------------------------------------

def top_answer(answer: dict):
    """(top choice, its probability) for any of Laya's three answer types."""
    if answer.get("type") == "noul" or "noul" in answer:
        p = float(answer["noul"])
        return (p >= 0.5), max(p, 1.0 - p)
    probabilities = answer["probabilities"]
    top = max(probabilities, key=probabilities.get)
    return top, float(probabilities[top])


def drift_between(base: Dict[str, dict], tuned: Dict[str, dict]) -> dict:
    """Share of items whose top answer changed, and the mean absolute change in the top
    probability (each model's own top choice). Both dicts are item id -> answer."""
    changed, delta = 0, []
    for item_id, before in base.items():
        b_top, b_p = top_answer(before)
        t_top, t_p = top_answer(tuned[item_id])
        changed += b_top != t_top
        delta.append(abs(b_p - t_p))
    n = len(base)
    return {"n_items": n, "changed": changed, "changed_share": round(changed / n, 4),
            "mean_abs_top_probability_change": round(sum(delta) / n, 4)}


def answers_on(agent, corpus: Corpus, item_ids: Sequence[str], questions: Dict[str, dict]
               ) -> Dict[str, Dict[str, dict]]:
    """question name -> item id -> answer, through ``Agent.system_one`` (shipped temperatures
    included, exactly what the flywheel would be served)."""
    out: Dict[str, Dict[str, dict]] = {name: {} for name in questions}
    for item_id in item_ids:
        answers = agent.system_one(corpus.items[item_id].text, questions)["answers"]
        for name in questions:
            out[name][item_id] = answers[name]
    return out


def drift_rows(corpus: Corpus, seed: int, lr: float, base: Dict[str, Dict[str, dict]],
               questions: Dict[str, dict]) -> List[dict]:
    """Fine-tune once with Arm A's recipe and compare every question's answers with base Laya's
    on paper600. Base answers are recomputed here in fp32 with the same code path, not read from
    ``fixtures/answers-laya.jsonl.gz`` (generated in fp16), so dtype is not mistaken for drift."""
    ids = corpus.recorded_ids
    agent = fresh_agent()
    train_laya(agent, "full", [corpus.items[i].text for i in ids],
               [corpus.label_index(i, "recorded") for i in ids], corpus.sentiment_question,
               epochs=epochs_for(len(ids)), lr=lr, seed=seed)
    agent.model.eval()
    tuned = answers_on(agent, corpus, corpus.paper600, questions)
    del agent
    return [{"arm": "A", "seed": seed, "n_labels": len(ids), "lr": lr, "question": name,
             "question_type": questions[name]["type"], **drift_between(base[name], tuned[name])}
            for name in questions]


# ---- driver ------------------------------------------------------------------------------------

def append_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(args) -> None:
    global CHECKPOINT_DIR, PROBS_DIR
    if args.save_checkpoints:
        CHECKPOINT_DIR = args.scratch / "checkpoints"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_probs:
        PROBS_DIR = args.scratch / "probs"
        PROBS_DIR.mkdir(parents=True, exist_ok=True)
    print("loading corpus (items, splits, recorded labels) ...")
    corpus = Corpus()
    cv_cache: Dict[Tuple[str, int], dict] = {}
    out = args.out
    total_started = time.time()

    if args.drift:
        print("\n=== M4: drift probe (Arm A, paper600) ===")
        questions = {"Sentiment": corpus.sentiment_question, **corpus.reference_questions,
                     **corpus.topic_question}
        lrs = {json.loads(line)["chosen_lr"] for line in out.read_text().splitlines()
               if json.loads(line)["arm"] == "A"} if out.exists() else set()
        if len(lrs) != 1:
            raise SystemExit(f"run Arm A first: expected one chosen lr in {out}, found {lrs}")
        lr = lrs.pop()
        agent = fresh_agent()
        base = answers_on(agent, corpus, corpus.paper600, questions)
        del agent
        rows: List[dict] = []
        for seed in args.seeds:
            print(f"seed {seed}: fine-tuning Arm A (lr={lr:g}) and asking every question again ...")
            new = drift_rows(corpus, seed, lr, base, questions)
            append_jsonl(args.drift_out, new)
            rows += new
            for r in new:
                print(f"  seed {seed} {r['question']:26} top answer changed {r['changed_share']:.3f}  "
                      f"mean |d top p| {r['mean_abs_top_probability_change']:.3f}", flush=True)
        print(f"\nwrote {len(rows)} rows to {args.drift_out}")
        return

    rows: List[dict] = []
    for arm in args.arms:
        if arm == "D":
            for seed in args.seeds:
                print(f"\n=== Arm D, seed {seed}, n=140 ===")
                row = run_student_point(corpus, seed, args.folds, cv_cache)
                rows.append(row)
                append_jsonl(out, [row])
                print(f"  paper600 acc {row['paper600_accuracy']:.3f}  full acc {row['full_accuracy']:.3f}  "
                      f"lr={row['chosen_lr']}  ({row['train_seconds']:.0f}s)")
        elif arm in ("A", "B"):
            for seed in args.seeds:
                print(f"\n=== Arm {arm}, seed {seed}, n=140 ===")
                row = run_laya_point(corpus, arm, 140, seed, "recorded", args.folds, cv_cache)
                rows.append(row)
                append_jsonl(out, [row])
                print(f"  paper600 acc {row['paper600_accuracy']:.3f}  full acc {row['full_accuracy']:.3f}  "
                      f"lr={row['chosen_lr']}  ({row['train_seconds']:.0f}s)")
        elif arm == "C":
            sizes = sorted(args.sizes)
            anchors = sorted(a for a in args.cv_anchors if a in sizes) if args.cv_anchors else sizes
            for n in sizes:
                if n in anchors:
                    reuse_from = None            # run this size's own CV
                else:
                    smaller = [a for a in anchors if a < n]
                    reuse_from = max(smaller) if smaller else min(anchors)
                for seed in args.seeds:
                    print(f"\n=== Arm C, seed {seed}, n={n} ===")
                    row = run_laya_point(corpus, "C", n, seed, "random-pool", args.folds, cv_cache,
                                         reuse_lr_from=reuse_from)
                    rows.append(row)
                    append_jsonl(out, [row])
                    print(f"  paper600 acc {row['paper600_accuracy']:.3f}  full acc "
                          f"{row['full_accuracy']:.3f}  lr={row['chosen_lr']}  "
                          f"({row['train_seconds']:.0f}s)")
        else:
            raise ValueError(f"unknown arm {arm!r}")

    print(f"\nwrote {len(rows)} rows to {out}; total wall time "
          f"{(time.time() - total_started) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=["A", "B", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sizes", type=int, nargs="+", default=[140, 300, 500, 800, 2000, 5140],
                        help="Arm C only")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--cv-anchors", type=int, nargs="*", default=[],
                        help="Arm C only: sizes at which to actually run CV; other sizes reuse "
                             "the nearest smaller anchor's lr (default: every size runs its own "
                             "CV, as pre-registered; pass e.g. 140 2000 to cut wall-clock further, "
                             "matching the pre-registration's own allowance to reuse n=2000's lr "
                             "at n=5140)")
    parser.add_argument("--out", type=Path, default=Path("studies/finetune_laya.jsonl"))
    parser.add_argument("--scratch", type=Path, default=Path("var/finetune_laya"))
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="spot-check only; writes under --scratch, never committed")
    parser.add_argument("--save-probs", action="store_true",
                        help="write per-item held-out (item id, truth, calibrated P(positive)) "
                             "to --scratch/probs/, for the selective-prediction comparison; "
                             "never under studies/, never committed")
    parser.add_argument("--drift", action="store_true", help="run the M4 drift probe instead")
    parser.add_argument("--drift-out", type=Path, default=Path("studies/finetune_laya_drift.jsonl"))
    main(parser.parse_args())

