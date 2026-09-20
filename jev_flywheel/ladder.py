"""The capability ladder: what the labels collected so far can support.

Model class and calibration method are both limited by how much evidence there is,
so they are one policy. As labels accumulate a scorecard climbs this ladder on its
own -- nobody has to notice that it crossed a threshold and edit a file.

The policy is a frozen, named profile, in the same idiom Plexus uses for its
optimization policy: ``CAPABILITY_LADDER_V1`` never changes in place, so a fit that
records "tier standard under capability-ladder-v1" stays interpretable after the
thresholds are revised in a v2.

Everything is gated on **effective** sample size (see ``sampling.kish_n_effective``),
never on the raw number of labels. Feedback that has been inverse-probability
weighted can have a handful of rows carrying most of the weight, so ten labels
may be worth three.

Two rules keep the ladder honest:

* **Permitted is not chosen.** The ladder says what the data can *support*. Whether
  a candidate is actually *better* is decided by out-of-fold metrics against the
  incumbent, and promotion requires beating it. Climbing on sample size alone is how
  you ship a fancier model that is worse than the simple one it replaced.
* **Descending is legal.** If a window shrinks or labels are invalidated, the
  effective sample size can fall. The fitter drops a tier rather than refit an
  over-parameterized model on thinner data, and reports that it did.

What this ladder does *not* include: generalized additive models, factorization
machines, boosted trees, neural networks and set encoders. Those are the further
rungs, they need artifact storage for their weights, and that is where Plexus's
model registry picks up. The measured reason to want them is in the lab notes:
boosting overtakes the logistic head at roughly 500 labels.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

LADDER_NAME = "capability-ladder-v1"


@dataclass(frozen=True)
class Tier:
    """One rung."""

    name: str
    min_n_effective: float
    model: str
    calibration: str
    # Regularization strengths tried by cross-validation, weakest last. Small C is
    # heavy shrinkage toward zero weights, which is what a small sample needs.
    c_grid: Tuple[float, ...]
    # A model may carry at most n_effective / features_per_n_effective features.
    # Heavier shrinkage tolerates more features per unit of evidence.
    features_per_n_effective: float
    note: str = ""

    def feature_budget(self, n_effective: float) -> int:
        if self.features_per_n_effective <= 0:
            return 0
        return max(int(n_effective // self.features_per_n_effective), 0)


CAPABILITY_LADDER_V1: Tuple[Tier, ...] = (
    Tier(
        name="hold", min_n_effective=0,
        model="linear_threshold", calibration="none", c_grid=(),
        features_per_n_effective=0,
        note="Too little evidence to fit anything. Keep the incumbent, hand-written "
             "or otherwise, and keep collecting labels.",
    ),
    Tier(
        name="shrunk", min_n_effective=30,
        model="multinomial_logistic", calibration="temperature",
        c_grid=(0.03, 0.1, 0.3), features_per_n_effective=5,
        note="Heavily regularized logistic head. One-parameter calibration only: "
             "isotonic memorizes at this size.",
    ),
    Tier(
        name="standard", min_n_effective=200,
        model="multinomial_logistic", calibration="temperature",
        c_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0), features_per_n_effective=10,
        note="Logistic head with regularization chosen by cross-validation. "
             "Temperature scaling still: the head is already calibrated on its own "
             "training distribution, and isotonic on top was measured to add noise.",
    ),
    Tier(
        name="rich", min_n_effective=1000,
        model="multinomial_logistic", calibration="two_stage",
        c_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0), features_per_n_effective=10,
        note="Enough labels for isotonic to correct what temperature cannot.",
    ),
)


def tier_for(n_effective: float, ladder: Tuple[Tier, ...] = CAPABILITY_LADDER_V1) -> Tier:
    """The richest tier the evidence supports."""
    chosen = ladder[0]
    for tier in ladder:
        if n_effective >= tier.min_n_effective:
            chosen = tier
    return chosen


def next_tier(n_effective: float,
              ladder: Tuple[Tier, ...] = CAPABILITY_LADDER_V1) -> Optional[Tier]:
    """The next rung up, or None at the top."""
    current = tier_for(n_effective, ladder)
    index = ladder.index(current)
    return ladder[index + 1] if index + 1 < len(ladder) else None


def distance_to_next(n_effective: float,
                     ladder: Tuple[Tier, ...] = CAPABILITY_LADDER_V1) -> Optional[float]:
    """How much more effective evidence the next rung needs."""
    upcoming = next_tier(n_effective, ladder)
    return None if upcoming is None else max(upcoming.min_n_effective - n_effective, 0.0)


class LadderRefusal(ValueError):
    """The evidence cannot support what was asked for. The message says what would."""
