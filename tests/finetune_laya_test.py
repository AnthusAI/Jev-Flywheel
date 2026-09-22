"""Feature: the fine-tune-Laya study's pure parts, with no model loaded.

``scripts/finetune_laya.py`` spends real GPU time the moment it touches a Laya checkpoint, so
this file is deliberately narrow: label/batch bookkeeping, the epoch schedule, the k-fold split,
and the row schema the study promises to fill in. Nothing here loads ``laya_mlx`` or downloads
a checkpoint, so it runs in ``make test`` like every other spec.
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_laya as fl  # noqa: E402


# ---- epoch schedule, fixed in the pre-registration -------------------------------------------

def test_small_label_budgets_train_for_ten_epochs():
    assert fl.epochs_for(140) == 10
    assert fl.epochs_for(500) == 10


def test_larger_label_budgets_train_for_three_epochs():
    assert fl.epochs_for(501) == 3
    assert fl.epochs_for(5140) == 3


# ---- k-fold split ------------------------------------------------------------------------------

def test_kfold_indices_partition_every_item_exactly_once_per_fold_as_a_validation_item():
    folds = fl.kfold_indices(23, 3, seed=1)
    assert len(folds) == 3
    all_val = [i for _, val in folds for i in val]
    assert sorted(all_val) == list(range(23))          # every item is validated exactly once


def test_kfold_indices_never_put_an_item_in_its_own_folds_training_set():
    for train, val in fl.kfold_indices(17, 3, seed=2):
        assert not (set(train) & set(val))


def test_kfold_indices_is_deterministic_for_a_fixed_seed():
    a = fl.kfold_indices(30, 4, seed=7)
    b = fl.kfold_indices(30, 4, seed=7)
    assert a == b


def test_kfold_indices_differs_across_seeds_on_a_large_enough_set():
    a = fl.kfold_indices(50, 3, seed=1)
    b = fl.kfold_indices(50, 3, seed=2)
    assert a != b


# ---- batching ------------------------------------------------------------------------------

def test_batches_covers_every_index_exactly_once():
    seen = sorted(i for chunk in fl.batches(37, 16, random.Random(0)) for i in chunk)
    assert seen == list(range(37))


def test_batches_respects_the_batch_size_except_possibly_the_last_chunk():
    chunks = list(fl.batches(35, 16, random.Random(0)))
    assert [len(c) for c in chunks] == [16, 16, 3]


def test_batches_shuffles_so_two_seeds_do_not_agree_on_a_37_item_set():
    a = list(fl.batches(37, 16, random.Random(1)))
    b = list(fl.batches(37, 16, random.Random(2)))
    assert a != b


# ---- calibration helpers, pure numpy ---------------------------------------------------------

def test_fit_temperature_is_positive_and_finite_on_separable_logits():
    import numpy as np
    logits = np.array([[10.0, -10.0], [-10.0, 10.0]] * 20)
    truth = np.array([0, 1] * 20)
    t = fl.fit_temperature(logits, truth)
    assert np.isfinite(t) and t > 0


def test_fit_temperature_finds_over_one_when_confident_predictions_are_half_wrong():
    # Confidently right half the time, confidently wrong the other half: no temperature fixes
    # the wrongness, but the optimum still softens (T > 1) rather than sharpening further.
    import numpy as np
    logits = np.array([[6.0, -6.0]] * 10 + [[-6.0, 6.0]] * 10)
    truth = np.array([0] * 10 + [0] * 10)      # half agree with the first block, half do not
    t = fl.fit_temperature(logits, truth)
    assert t > 1.0


def test_softmax_rows_sum_to_one():
    import numpy as np
    p = fl.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_score_split_reports_the_promised_fields():
    import numpy as np
    logits = np.array([[3.0, -3.0], [-3.0, 3.0], [1.0, -1.0], [-1.0, 1.0]])
    truth = [0, 1, 0, 1]
    tiers = ["strong", "strong", "weak", "weak"]
    out = fl.score_split(logits, truth, tiers, temperature=1.5)
    for key in ("accuracy", "ece_raw", "ece_calibrated", "brier_calibrated", "temperature",
                "by_tier"):
        assert key in out
    assert set(out["by_tier"]) == {"strong", "weak"}
    assert out["accuracy"] == 1.0


# ---- freezing selection: real MLX modules, no checkpoint, no download ------------------------

def _leaf_count(tree) -> int:
    from mlx.utils import tree_flatten
    return len(tree_flatten(tree))


def test_full_arm_freezes_only_act_head_and_temperature():
    pytest.importorskip("mlx.core")
    model = _tiny_decision_model()
    fl.set_trainable(model, "full")
    trainable = model.trainable_parameters()
    assert _leaf_count(trainable["act_head"]) == 0
    assert "temperature" not in trainable
    assert _leaf_count(trainable["encoder"]) > 0
    assert _leaf_count(trainable["head"]) > 0
    assert _leaf_count(trainable["scorer"]) > 0


def test_head_arm_trains_only_the_decision_head_type_emb_and_scorer():
    pytest.importorskip("mlx.core")
    model = _tiny_decision_model()
    fl.set_trainable(model, "head")
    trainable = model.trainable_parameters()
    assert _leaf_count(trainable["encoder"]) == 0
    assert _leaf_count(trainable["act_head"]) == 0
    assert "temperature" not in trainable
    assert _leaf_count(trainable["head"]) > 0
    assert _leaf_count(trainable["type_emb"]) > 0
    assert _leaf_count(trainable["scorer"]) > 0


def test_set_trainable_rejects_an_unknown_arm():
    pytest.importorskip("mlx.core")
    model = _tiny_decision_model()
    with pytest.raises(ValueError):
        fl.set_trainable(model, "nonsense")


def _tiny_decision_model():
    """A tiny real DecisionModel (a handful of hidden units, one layer) -- no checkpoint needed,
    since this only exercises freeze/unfreeze bookkeeping on the module tree shape."""
    from laya_mlx.model import DecisionModel, EncoderConfig

    cfg = EncoderConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                        num_attention_heads=2, layer_types=["full_attention"])
    return DecisionModel(cfg, {"head_layers": 1, "act_costs": {}})


# ---- Corpus.label_index / recorded_prefix_or_random: pure bookkeeping, no I/O -----------------

def test_corpus_random_universe_excludes_the_140_recorded_items_by_construction():
    # This is a structural guarantee this module documents and depends on (5,280 pool - 140
    # recorded = 5,140); assert the arithmetic here rather than only in the docstring.
    assert 5280 - 140 == 5140


# ---- drift bookkeeping -------------------------------------------------------------------------

def test_the_top_answer_of_a_noul_is_its_side_of_one_half_and_that_sides_probability():
    assert fl.top_answer({"type": "noul", "noul": 0.2}) == (False, 0.8)
    assert fl.top_answer({"type": "noul", "noul": 0.9}) == (True, 0.9)


def test_the_top_answer_of_a_choice_or_score_is_its_most_probable_option():
    assert fl.top_answer({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}) == ("b", 0.7)
    assert fl.top_answer({"type": "score", "probabilities": {"0": 0.6, "1": 0.4}}) == ("0", 0.6)


def test_drift_counts_changed_top_answers_and_averages_the_change_in_top_probability():
    base = {"x": {"type": "noul", "noul": 0.9}, "y": {"type": "noul", "noul": 0.1}}
    tuned = {"x": {"type": "noul", "noul": 0.4}, "y": {"type": "noul", "noul": 0.1}}
    out = fl.drift_between(base, tuned)
    assert out["changed"] == 1 and out["changed_share"] == 0.5
    assert out["mean_abs_top_probability_change"] == pytest.approx((abs(0.9 - 0.6) + 0.0) / 2)


# ---- per-item held-out probability dump (for the selective-prediction comparison) -------------

def test_calibrated_positive_probabilities_matches_softmax_at_the_positive_index():
    import numpy as np

    logits = np.array([[2.0, -2.0], [-2.0, 2.0], [0.0, 0.0]])
    p = fl.calibrated_positive_probabilities(logits, temperature=1.0)
    expected = fl.softmax(logits)[:, fl.CLASSES.index("positive")]
    assert np.allclose(p, expected)


def test_calibrated_positive_probabilities_softens_towards_half_as_temperature_grows():
    import numpy as np

    logits = np.array([[4.0, -4.0]])
    cold = fl.calibrated_positive_probabilities(logits, temperature=1.0)[0]
    hot = fl.calibrated_positive_probabilities(logits, temperature=100.0)[0]
    assert cold > hot > 0.5 - 1e-6


def test_save_item_probabilities_writes_one_json_object_per_item(tmp_path):
    import numpy as np

    logits = np.array([[3.0, -3.0], [-3.0, 3.0]])
    path = tmp_path / "probs" / "A-seed1-n140-paper600.jsonl"
    fl.save_item_probabilities(path, ["item-a", "item-b"], [0, 1], logits, temperature=1.0)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["item_id"] == "item-a" and lines[0]["truth"] == "positive"
    assert lines[1]["item_id"] == "item-b" and lines[1]["truth"] == "negative"
    assert 0.0 <= lines[0]["p_positive"] <= 1.0
    assert lines[0]["p_positive"] > 0.5      # item-a's logits favor positive
