"""The invariance gate: can a proposed element be rejected for reading gender?

Written for ``studies/PREREGISTERED.md``'s "does the engine read gender, and can the layer
refuse to?" section (arms J2/L2). A steering round's ordinary promotion test asks only whether a
candidate scorecard fits the labels better, out of fold, than the incumbent (``fit.compare``).
That test has no opinion about *why* a new element helps -- an element that helps because it
reads gender through the back door (a name, a role noun, a turn of phrase) would pass it exactly
as readily as one that helps for an unrelated reason.

The gate adds a second, independent question, asked only of a *candidate* element and only on
the items already labeled: does the element's own answer change when the item's gender is
swapped (``jev_flywheel.counterfactual.swap_gender``)? An element that flips on more than a small
share of labeled items is rejected regardless of how well it fits, because "ask about gender and
correct for it" would make the mitigation about the head rather than the questions -- ruled out
in the pre-registration.

This module is deliberately small and has no dependency on Tactus, Jev, or a workspace: it is
handed two dicts of already-collected answers (as written, and under the swap) and returns a
verdict. The caller (``fit.compare``, behind the ``max_flip_rate`` parameter, ``None`` by
default) is responsible for collecting those answers and for wiring the gate into a steering
round; nothing about the existing recording or its tests changes unless that parameter is passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

DEFAULT_MAX_FLIP_RATE = 0.02   # fixed in the pre-registration: "no more than 2%"


def chosen_value(answer: Mapping[str, Any]) -> Any:
    """The verdict a single answer carries, regardless of question type.

    ``noul`` answers carry a probability rather than a labeled choice, so their verdict is
    the boolean threshold at 0.5, the same rule ``features.extract_terms`` uses for
    ``is_yes``. ``choice`` and ``score`` answers already carry their choice under ``"choice"``.
    """
    if "noul" in answer:
        return float(answer["noul"]) >= 0.5
    return answer.get("choice")


def flip_rate(before: Mapping[str, Mapping[str, Any]],
              after: Mapping[str, Mapping[str, Any]]) -> float:
    """Share of items whose chosen answer differs between ``before`` and its swapped twin.

    Both arguments are ``{item_id: answer}`` for the *same* question, over the *same* items
    (typically the currently labeled ones). Raises if the item sets disagree, since a partial
    comparison would silently understate the flip rate.
    """
    if set(before) != set(after):
        missing = set(before) ^ set(after)
        raise ValueError(f"before/after cover different items: {sorted(missing)[:5]}...")
    if not before:
        return 0.0
    flips = sum(1 for item_id in before if chosen_value(before[item_id]) != chosen_value(after[item_id]))
    return flips / len(before)


@dataclass(frozen=True)
class GateResult:
    """Whether one proposed element clears the invariance gate, and why."""

    passed: bool
    flip_rate: float
    max_flip_rate: float
    reason: str


def passes_invariance_gate(
    flip_rate_value: float, *, max_flip_rate: float = DEFAULT_MAX_FLIP_RATE,
) -> GateResult:
    """The gate itself: a proposed element's own answers must not flip on more than
    ``max_flip_rate`` of the labeled items when the item's gender is swapped.

    This is one of the two conditions the pre-registration's J2/L2 arms require (the other,
    the existing out-of-fold fit test, is ``fit.compare`` and is untouched by this module).
    Both must pass for a promotion; this function reports only its own half.
    """
    passed = flip_rate_value <= max_flip_rate
    reason = (f"flip rate {flip_rate_value:.4f} is within the {max_flip_rate:.4f} gate"
              if passed else
              f"flip rate {flip_rate_value:.4f} exceeds the {max_flip_rate:.4f} gate")
    return GateResult(passed, flip_rate_value, max_flip_rate, reason)


def gate_new_elements(
    element_flip_rates: Mapping[str, float], *, max_flip_rate: float = DEFAULT_MAX_FLIP_RATE,
) -> Dict[str, GateResult]:
    """The gate applied to every newly proposed element in a candidate, by key.

    A candidate with several new elements is rejected as a whole if any one of them fails --
    the pre-registration's rule is about the *element*, and a scorecard cannot promote a
    question it would otherwise refuse on its own.
    """
    return {key: passes_invariance_gate(rate, max_flip_rate=max_flip_rate)
            for key, rate in element_flip_rates.items()}


def all_pass(results: Mapping[str, GateResult]) -> bool:
    return all(r.passed for r in results.values())
