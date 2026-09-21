#!/usr/bin/env python
"""Distil the calibrated head into a small local text classifier, and decide whether it may serve.

    pip install torch transformers          # the "student" extra; ~80 MB + the model (~270 MB)
    python scripts/distill_student.py --scratch /tmp/lc     # after scripts/learning_curve.py --prepare

The process, end to end, so it can be repeated as the teacher improves:

1. **Teacher.** The repo's own head, fitted by ``fit_head`` on the recorded *human* labels (Jev's
   holistic answer, the seven cached elements, and the discovered topic element), labels every
   pool item with a calibrated probability. No human label is used for anything else.
2. **Student.** A fine-tuned ``AutoModelForSequenceClassification`` (DistilBERT by default) reads
   the raw text and nothing else. Three are trained per seed: on the teacher's *soft* labels (its
   probabilities), on its *hard* labels, and on the corpus's reference labels as the ceiling ("if
   we had a human label for every pool item"). The items the human labeled are **held out of
   student training** so they can calibrate it.
3. **Calibrate.** One temperature, fitted by maximum likelihood on the human-labeled items the
   student never saw. (An uncalibrated student is over-confident, which would make any
   confidence threshold meaningless.)
4. **Evaluate.** On held-out items no one trained on: accuracy against the human label, agreement
   with the teacher, ECE before and after calibration, and every (tier, topic) slice.
5. **Gate.** The student may serve a slice only if it is within ``MARGIN`` of the teacher against
   the human label there, on at least ``MIN_SLICE`` items. The gate is per slice on purpose: an
   overall number hides the slice a student is bad at.
6. **Cascade.** The student answers when its calibrated confidence clears a threshold and the
   teacher takes the rest. Coverage and accuracy are reported per threshold.

Evaluating a student against the *teacher's* labels alone would be circular, so the headline is
always against the human label. Held-out items are never trained on and never label the teacher.
Results go to one JSONL, a row per (student, seed).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from student_proxy import build_teacher  # noqa: E402

from jev_flywheel.evaluate import summarize  # noqa: E402
from jev_flywheel.items import FeedbackItem, JsonlStore  # noqa: E402
from learning_curve import RECORDING  # noqa: E402

MARGIN = 0.02             # a slice passes if the student is within this of the teacher
MIN_SLICE = 30            # and there are at least this many held-out human-labeled items in it
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95)
CLASSES = ("positive", "negative")


def train_student(model_name, texts, targets, *, epochs, lr, batch, seed, device, max_len):
    """Fine-tune on ``targets``: an (n, 2) array of probabilities (one-hot for hard labels)."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts) + batch - 1) // batch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    targets = torch.tensor(targets, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(texts))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits
            loss = -(targets[idx].to(device) * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
    model.eval()
    return tok, model


def logits_for(tok, model, texts, *, device, max_len, batch=128):
    import numpy as np
    import torch
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            enc = tok(texts[start:start + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out)


def fit_temperature(logits, truth):
    """The single temperature that minimises log loss on ``truth`` (class indices)."""
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


def latency_ms(tok, model, texts, device, max_len, n=100):
    import torch
    samples = []
    with torch.no_grad():
        for text in texts[:n]:
            t0 = time.perf_counter()
            enc = tok([text], truncation=True, max_length=max_len, return_tensors="pt").to(device)
            model(**enc).logits.float().cpu()
            samples.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(samples[10:]), 2)


def main(args) -> None:
    import numpy as np
    import torch

    items, pool, test, t_pool, t_test, fit = build_teacher(args.scratch)
    recorded = JsonlStore(RECORDING / "feedback.jsonl", FeedbackItem).all()
    human = {f.item_id: f.final_answer_value for f in recorded}
    train_ids = [i for i in pool if i not in human]            # the human-labeled items are held out
    cal_ids = [i for i in human if i in items]
    print(f"student trains on {len(train_ids)} teacher-labeled items; calibrates on the "
          f"{len(cal_ids)} human-labeled items it never sees; is scored on {len(test)} held-out items")

    def p_positive(record):
        label, confidence, _ = record
        return confidence if label == "positive" else 1.0 - confidence

    soft = np.array([[p_positive(t_pool[i]), 1 - p_positive(t_pool[i])] for i in train_ids])
    hard = np.array([[1.0, 0.0] if t_pool[i][0] == "positive" else [0.0, 1.0] for i in train_ids])
    ref = np.array([[1.0, 0.0] if items[i].reference_label == "positive" else [0.0, 1.0]
                    for i in train_ids])
    kinds = {"soft-teacher": soft, "hard-teacher": hard, "reference-ceiling": ref}
    texts = [items[i].text for i in train_ids]
    test_texts = [items[i].text for i in test]
    cal_texts = [items[i].text for i in cal_ids]
    index = {c: k for k, c in enumerate(CLASSES)}
    truth = np.array([index[items[i].reference_label] for i in test])
    cal_truth = np.array([index[human[i] if human[i] in index else items[i].reference_label]
                          for i in cal_ids])
    teacher_pred = np.array([index[t_test[i][0]] for i in test])
    tiers = np.array([items[i].metadata["tier"] for i in test])
    topics = np.array([t_test[i][2] for i in test])
    teacher_correct = teacher_pred == truth
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []

    for seed in args.seeds:
        for kind, targets in kinds.items():
            started = time.time()
            tok, model = train_student(args.model, texts, targets, epochs=args.epochs, lr=args.lr,
                                       batch=args.batch, seed=seed, device=device,
                                       max_len=args.max_len)
            trained = time.time() - started
            lt = logits_for(tok, model, test_texts, device=device, max_len=args.max_len)
            lc = logits_for(tok, model, cal_texts, device=device, max_len=args.max_len)
            temperature = fit_temperature(lc, cal_truth)
            raw_p, cal_p = softmax(lt), softmax(lt / temperature)
            pred = raw_p.argmax(axis=1)
            correct = pred == truth
            confidence = cal_p.max(axis=1)
            row = {
                "student": kind, "seed": seed, "model": args.model, "epochs": args.epochs,
                "train_items": len(train_ids), "train_seconds": round(trained, 1),
                "temperature": round(temperature, 3),
                "acc_vs_human": round(float(correct.mean()), 4),
                "agree_with_teacher": round(float((pred == teacher_pred).mean()), 4),
                "teacher_acc_vs_human": round(float(teacher_correct.mean()), 4),
                "ece_raw": round(summarize(raw_p.max(axis=1).tolist(), correct.astype(int).tolist()).ece, 4),
                "ece_calibrated": round(summarize(confidence.tolist(), correct.astype(int).tolist()).ece, 4),
                "ms_per_item_batch1_" + device: latency_ms(tok, model, test_texts, device, args.max_len),
                "slices": {}, "cascade": {},
            }
            for tier in sorted(set(tiers)):
                for topic in sorted(set(topics)):
                    m = (tiers == tier) & (topics == topic)
                    if m.sum() >= MIN_SLICE:
                        s, t = float(correct[m].mean()), float(teacher_correct[m].mean())
                        row["slices"][f"{tier}/{topic}"] = {
                            "n": int(m.sum()), "student": round(s, 4), "teacher": round(t, 4),
                            "passes_gate": bool(s >= t - MARGIN)}
            for thr in THRESHOLDS:
                m = confidence >= thr
                cascade = np.where(m, pred, teacher_pred) == truth
                row["cascade"][str(thr)] = {
                    "coverage": round(float(m.mean()), 4),
                    "student_acc_where_it_answers": round(float(correct[m].mean()), 4) if m.any() else None,
                    "teacher_acc_on_the_same_items": round(float(teacher_correct[m].mean()), 4) if m.any() else None,
                    "cascade_acc": round(float(cascade.mean()), 4)}
            rows.append(row)
            print(f"seed {seed} {kind:18} vs human {row['acc_vs_human']:.3f}  "
                  f"vs teacher {row['agree_with_teacher']:.3f}  ECE {row['ece_raw']:.3f} -> "
                  f"{row['ece_calibrated']:.3f}  T={temperature:.2f}  "
                  f"{[v for k, v in row.items() if k.startswith('ms_per')][0]} ms  "
                  f"({trained:.0f}s train)", flush=True)
            del model
            if device == "mps":
                torch.mps.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nteacher accuracy vs human on the held-out set: {float(teacher_correct.mean()):.3f}")
    for kind in kinds:
        own = [r for r in rows if r["student"] == kind]
        acc = [r["acc_vs_human"] for r in own]
        print(f"{kind:18} vs human {statistics.mean(acc):.3f} "
              f"(min {min(acc):.3f}, max {max(acc):.3f}, {len(acc)} seeds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=Path("studies/distill.jsonl"))
    main(parser.parse_args())
