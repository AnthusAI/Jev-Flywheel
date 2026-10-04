"""Small Pareto pool for question-set experiments.

The pool only compares validation summaries. It never sees held-out observations, and
question-set merges go through the ordinary proposal application path so fitted numbers
are cleared and must be learned again.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping

from jev_flywheel.proposal import ElementAdd, Proposal, apply_proposal
from jev_flywheel.scorecard import Scorecard


JEV_REQUEST_CAP = 2_000


@dataclass(frozen=True)
class AnalystCall:
    """One fixed slot in the matched-budget adaptive-search schedule."""

    index: int
    phase: str


@dataclass(frozen=True)
class AdaptiveSearchPlan:
    """Deterministic analyst-call schedule and shared Jev request ceiling."""

    calls: tuple[AnalystCall, ...]
    jev_request_cap: int = JEV_REQUEST_CAP


def adaptive_search_plan() -> AdaptiveSearchPlan:
    """Return four common discovery calls, then two greedy and two Pareto calls."""
    phases = ("shared_initial",) * 4 + ("greedy",) * 2 + ("pareto",) * 2
    return AdaptiveSearchPlan(tuple(AnalystCall(i + 1, phase)
                                    for i, phase in enumerate(phases)))


def has_distinct_slice_specialists(
    candidates: Iterable[Candidate], *, min_slice_effective_n: float = 0,
) -> bool:
    """Whether at least two candidates uniquely win different predefined slices.

    A Pareto search only earns its mechanism claim when retained candidates specialize
    in different slices. Ties do not count as a distinct specialist.
    """
    if min_slice_effective_n < 0:
        raise ValueError("min_slice_effective_n cannot be negative")
    eligible = [c for c in candidates if _valid(c, min_slice_effective_n)]
    if len(eligible) < 2:
        return False
    slices = set(eligible[0].slice_brier)
    if not slices or any(set(c.slice_brier) != slices for c in eligible):
        return False
    winners = set()
    for name in slices:
        best = min(c.slice_brier[name] for c in eligible)
        tied = [c for c in eligible if c.slice_brier[name] == best]
        if len(tied) == 1:
            winners.add(tied[0].candidate_id)
    return len(winners) >= 2


def fresh_split_ids(
    pool_item_ids: Iterable[str], *, excluded_ids: Iterable[str], seed: int,
    selection_size: int, test_size: int,
) -> tuple[list[str], list[str]]:
    """Freeze disjoint selection and test IDs after all prior study IDs are removed."""
    if selection_size < 1 or test_size < 1:
        raise ValueError("selection_size and test_size must be positive")
    excluded = set(excluded_ids)
    eligible = sorted(set(pool_item_ids) - excluded)
    if len(eligible) < selection_size + test_size:
        raise ValueError(f"only {len(eligible)} fresh pool IDs remain; need "
                         f"{selection_size + test_size}")
    random.Random(seed).shuffle(eligible)
    return eligible[:selection_size], eligible[selection_size:selection_size + test_size]


@dataclass(frozen=True)
class Candidate:
    """A fitted scorecard with validation summaries for JEVPA selection."""

    candidate_id: str
    scorecard: Scorecard
    slice_brier: Mapping[str, float]
    slice_effective_n: Mapping[str, float]
    overall_brier: float
    overall_eligible: bool = True


def _valid(candidate: Candidate, min_slice_effective_n: float) -> bool:
    if not candidate.overall_eligible or not candidate.slice_brier:
        return False
    if set(candidate.slice_brier) != set(candidate.slice_effective_n):
        return False
    return all(
        candidate.slice_effective_n[slice_name] >= min_slice_effective_n
        and 0.0 <= score <= 1.0
        for slice_name, score in candidate.slice_brier.items()
    ) and 0.0 <= candidate.overall_brier <= 1.0


def _dominates(left: Candidate, right: Candidate) -> bool:
    """Whether left is no worse on every shared slice and strictly better on one."""
    if set(left.slice_brier) != set(right.slice_brier):
        return False
    left_values = [left.slice_brier[name] for name in sorted(left.slice_brier)]
    right_values = [right.slice_brier[name] for name in sorted(right.slice_brier)]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def pareto_pool(
    candidates: Iterable[Candidate], *, max_size: int, min_slice_effective_n: float
) -> List[Candidate]:
    """Return a deterministic, bounded set of eligible nondominated candidates.

    Each slice is a separate minimization objective. Candidates must meet the minimum
    effective sample size on every reported slice, and eligible candidates must share
    the same predefined slice names. For a cap, the front is ordered by mean slice
    Brier and ID after reserving each slice's best candidate. The caller tracks
    the best overall scorecard separately.
    """
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    if min_slice_effective_n < 0:
        raise ValueError("min_slice_effective_n cannot be negative")
    by_id: Dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.candidate_id in by_id:
            raise ValueError(f"duplicate candidate_id {candidate.candidate_id!r}")
        by_id[candidate.candidate_id] = candidate
    eligible = [c for c in by_id.values() if _valid(c, min_slice_effective_n)]
    if not eligible:
        return []
    expected_slices = set(eligible[0].slice_brier)
    if any(set(candidate.slice_brier) != expected_slices for candidate in eligible[1:]):
        raise ValueError("eligible candidates must use the same predefined slices")
    front = [c for c in eligible if not any(_dominates(other, c) for other in eligible if other is not c)]
    front.sort(key=lambda c: (sum(c.slice_brier.values()) / len(c.slice_brier), c.candidate_id))
    chosen: List[Candidate] = []
    for slice_name in sorted(expected_slices):
        winner = min(front, key=lambda c: (c.slice_brier[slice_name], c.candidate_id))
        if winner not in chosen:
            chosen.append(winner)
        if len(chosen) == max_size:
            return chosen
    chosen.extend(candidate for candidate in front if candidate not in chosen)
    return chosen[:max_size]


def _element_signature(element) -> tuple:
    return (element.key, element.question_type, element.instructions, element.criteria)


def merge_question_sets(base: Scorecard, donor: Scorecard, score_name: str) -> Scorecard:
    """Merge compatible own questions from donor into base and clear fitted state.

    The union is represented as a normal ``Proposal`` and applied with
    ``apply_proposal``. Existing duplicate keys must have identical definitions.
    Shared questions and holistic wording must agree; they are not silently rewritten.
    """
    base_score = base.score(score_name)
    donor_score = donor.score(score_name)
    if base_score.decision is None or donor_score.decision is None:
        raise ValueError(f"score {score_name!r} needs a decision block in both scorecards")
    if (base_score.question_type, base_score.instructions, base_score.criteria,
            base_score.decision.classes) != (
            donor_score.question_type, donor_score.instructions, donor_score.criteria,
            donor_score.decision.classes):
        raise ValueError("scorecards have incompatible holistic questions or decision classes")
    if [_element_signature(e) for e in base_score.shared_elements] != [
            _element_signature(e) for e in donor_score.shared_elements]:
        raise ValueError("scorecards have incompatible shared questions")

    existing = {element.key: element for element in base_score.elements}
    additions = []
    for element in donor_score.elements:
        if element.key in existing:
            if _element_signature(existing[element.key]) != _element_signature(element):
                raise ValueError(f"element {element.key!r} has incompatible question definitions")
            continue
        prefix = element.key + "."
        features = [feature for feature in donor_score.decision.features
                    if feature.startswith(prefix)]
        additions.append(ElementAdd(
            key=element.key, question_type=element.question_type,
            instructions=element.instructions, criteria=element.criteria,
            features=features or None,
        ))
    proposal = Proposal(add=additions)
    try:
        merged = apply_proposal(base, score_name, proposal)
    except Exception as error:
        # Surface an intentional merge error without hiding its original explanation.
        raise ValueError(f"cannot merge question sets: {error}") from error
    return merged
