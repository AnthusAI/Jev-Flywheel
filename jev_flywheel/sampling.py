"""Undoing the bias that choosing which items to ask about creates.

The console does not show items at random. It shows the ones a labeler learns the
most from, which are disproportionately the hard, atypical ones. A head fit on
those labels as if they were a random sample learns a distorted world: it looks
excellent on the labeled items and does worse in production, because the labeled
items over-represent exactly the region where the previous model was wrong.

The correction is inverse-probability weighting. Every labeled item carries the
probability the selection rule gave it when it was shown, and the fit weights it
by the reciprocal, so an item that was unlikely to be picked stands in for the
many like it that were not. This is the same problem Plexus solves for feedback
sampled per confusion-matrix cell, and the same cure.

It only works if selection is *stochastic*. A rule that always shows the single
highest-scoring item gives that item probability one and every other item zero,
and there is nothing to invert. So selection samples from a distribution, and
records the probability it sampled with.

Weighting has a price, which is why every guard here is stated in **effective**
sample size and never in raw item counts. Ten labels where one carries weight 50
are not ten labels.
"""
from typing import List, Optional, Sequence


def inverse_propensity_weights(
    propensities: Sequence[float],
    *,
    max_ratio: Optional[float] = 20.0,
    normalize: bool = True,
) -> List[float]:
    """Weights of ``1 / propensity``, capped and rescaled to sum to ``n``.

    ``max_ratio`` caps any weight at that multiple of the median. An item with a
    tiny propensity would otherwise dominate the likelihood, and one noisy label
    would move the whole fit. Capping trades a little bias for a lot of variance,
    which is the right trade at feedback-set sizes. ``None`` disables it.

    Normalizing to ``sum == n`` keeps regularization strength comparable between
    a weighted and an unweighted fit; without it, changing the propensity scale
    would silently change how hard the model is regularized.
    """
    if any(p is None or p <= 0 or p > 1 for p in propensities):
        raise ValueError(
            "every labeled item needs a propensity in (0, 1]; a label with none "
            "cannot be weighted, so it must be dropped rather than guessed")
    weights = [1.0 / p for p in propensities]
    if max_ratio is not None and weights:
        ordered = sorted(weights)
        median = ordered[len(ordered) // 2]
        cap = median * max_ratio
        weights = [min(w, cap) for w in weights]
    if normalize and weights:
        scale = len(weights) / sum(weights)
        weights = [w * scale for w in weights]
    return weights


def kish_n_effective(weights: Sequence[float]) -> float:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``.

    Equal weights give ``n``. Unequal weights give less, because a few heavy rows
    dominate: a 3,140-item cell sampled at 50 carries weight 51.8 while a 5-item
    cell carries 1.0, and the effective size can be an order of magnitude below
    the row count. This is the number every capability gate reads.
    """
    if not weights:
        return 0.0
    total = sum(weights)
    squares = sum(w * w for w in weights)
    return (total * total) / squares if squares else 0.0


def selection_distribution(
    scores: Sequence[float],
    *,
    temperature: float = 1.0,
    explore: float = 0.1,
) -> List[float]:
    """Turn selection scores into a probability of showing each candidate.

    A softmax over the scores, mixed with a uniform floor. The floor is
    ``explore`` of the total mass, spread evenly, which does two jobs: it makes
    every propensity strictly positive, so every item can be inverse-weighted, and
    it keeps the labeled set from being *entirely* a product of the current
    model's idea of what is interesting.

    Low temperature approaches "always pick the best" and makes weights explode;
    high temperature approaches random sampling and wastes labels. The default is
    a compromise, and the explore floor bounds the worst case: no weight can
    exceed ``n / explore`` however sharp the softmax is.
    """
    n = len(scores)
    if n == 0:
        return []
    if not 0.0 < explore <= 1.0:
        raise ValueError("explore must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    top = max(scores)
    exps = [pow(2.718281828459045, (s - top) / temperature) for s in scores]
    total = sum(exps)
    return [(1.0 - explore) * e / total + explore / n for e in exps]
