"""When is a steering adjustment worth making?

There are two operations, and their economics are two orders of magnitude apart, so
they get two triggers:

* **Refit** re-estimates the head's weights from the labels so far. It takes
  milliseconds, needs no Jev calls and no LLM, and a candidate is only promoted if it
  beats the incumbent out of fold. It is cheap enough that the policy is essentially
  "try it every few labels, and promote if better".
* **Rethink** is the meta-cognition pass: a language model reads the mismatches and
  the human's comments and proposes changes to the elements themselves. It costs an
  LLM call, and if it proposes new elements, a Jev top-up over the labeled items. It
  should fire when the residual is *structured* rather than noisy -- when more labels
  would not help but better questions might.

That distinction is the heart of the trigger. A plateau in out-of-fold loss while
commented mismatches keep arriving says the feature set is the bottleneck, not the
sample size. Interrupting the human before then wastes their attention; waiting long
after wastes labels.

The policy is a pure function of a small state record, frozen and named like the other
policies in this project. It returns, for each trigger, every condition with whether it
is met and how far off it is, because the explanation is the demonstration: the system
saying why it is or is not worth interrupting you.

One thing the policy cannot judge is whether the new comments raise concepts the
current elements do not cover. That takes a language model, so it is the rethink
procedure's job, not this function's. The comment count is the cheap proxy.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jev_flywheel.ladder import CAPABILITY_LADDER_V1

POLICY_NAME = "steering-v1"


@dataclass(frozen=True)
class SteeringPolicy:
    name: str = POLICY_NAME
    refit_every: int = 5                        # new labels since the last fit attempt
    rethink_min_new_labels: int = 15            # cooldown between rethinks
    rethink_min_commented_mismatches: int = 5   # disagreements the human explained
    plateau_window: int = 3                     # refits over which to look for progress
    plateau_min_improvement: float = 0.01       # out-of-fold log loss, per window
    min_headroom: float = 0.03                  # 1 - out-of-fold accuracy


@dataclass
class SteeringState:
    """What the policy needs to know, derived from the workspace's history."""

    n_labeled: int = 0
    n_effective: float = 0.0
    labels_since_fit: int = 0
    labels_since_rethink: int = 0
    commented_mismatches_since_rethink: int = 0
    fit_log_losses: List[float] = field(default_factory=list)    # oldest first
    oof_accuracy: Optional[float] = None


@dataclass
class Condition:
    """One requirement of a trigger, with how close it is."""

    name: str
    met: bool
    have: str
    need: str


@dataclass
class Trigger:
    name: str
    fire: bool
    conditions: List[Condition]

    @property
    def unmet(self) -> List[Condition]:
        return [c for c in self.conditions if not c.met]

    @property
    def summary(self) -> str:
        if self.fire:
            return f"{self.name}: worth doing now"
        first = self.unmet[0]
        return f"{self.name}: not yet ({first.name}: have {first.have}, need {first.need})"


def _fit_floor() -> float:
    return CAPABILITY_LADDER_V1[1].min_n_effective


def evaluate(state: SteeringState, policy: SteeringPolicy = SteeringPolicy()) -> Dict[str, Trigger]:
    """Both triggers, each with its conditions laid out."""
    fit_floor = _fit_floor()
    can_fit = state.n_effective >= fit_floor

    refit = [
        Condition("enough evidence to fit", can_fit,
                  f"{state.n_effective:.0f} effective labels", f"{fit_floor:.0f}"),
        Condition("new labels since the last fit", state.labels_since_fit >= policy.refit_every,
                  str(state.labels_since_fit), str(policy.refit_every)),
    ]

    losses = state.fit_log_losses
    window = policy.plateau_window
    if len(losses) > window:
        improvement = losses[-1 - window] - losses[-1]
        plateau = improvement < policy.plateau_min_improvement
        plateau_have = f"{improvement:+.4f} loss gain over the last {window} refits"
    else:
        plateau = False
        plateau_have = f"{len(losses)} refit(s); too few to tell a plateau from early progress"
    headroom = None if state.oof_accuracy is None else 1.0 - state.oof_accuracy

    rethink = [
        Condition("a candidate can be evaluated", can_fit,
                  f"{state.n_effective:.0f} effective labels", f"{fit_floor:.0f}"),
        Condition("cooldown since the last rethink",
                  state.labels_since_rethink >= policy.rethink_min_new_labels,
                  f"{state.labels_since_rethink} new labels", str(policy.rethink_min_new_labels)),
        Condition("mismatches the human explained",
                  state.commented_mismatches_since_rethink >= policy.rethink_min_commented_mismatches,
                  str(state.commented_mismatches_since_rethink),
                  str(policy.rethink_min_commented_mismatches)),
        Condition("progress has plateaued", plateau, plateau_have,
                  f"under {policy.plateau_min_improvement} per {window} refits"),
        Condition("room left to improve",
                  headroom is not None and headroom >= policy.min_headroom,
                  "unknown" if headroom is None else f"{headroom:.3f} error",
                  f"at least {policy.min_headroom}"),
    ]
    return {
        "refit": Trigger("refit", all(c.met for c in refit), refit),
        "rethink": Trigger("rethink", all(c.met for c in rethink), rethink),
    }
