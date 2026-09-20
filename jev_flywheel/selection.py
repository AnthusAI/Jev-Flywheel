"""Active selection: which item is worth asking the human about next.

Labels are the scarce resource, so each one should teach the most it can. This is
the twenty-questions part of the design: instead of labeling items in file order,
the console asks about the item whose answer is most informative, given everything
the system currently believes.

Five signals, each cheap and each computed from what is already on hand:

* **Uncertainty** -- the head's own confidence is near a coin flip. The classic
  signal, and the one that on its own walks straight into a trap (below).
* **Disagreement** -- the head and Jev's holistic answer say different things. Free,
  because both are already computed. These are exactly the items where the elements
  are doing the work, so a label adjudicates whether the override was right.
* **Conflict** -- the evidence pulls both ways: a large share of the weighted
  contributions argue against the decision the head made. These are the items whose
  human comments most often name a factor nobody declared.
* **Novelty** -- far from every item already labeled, so the human is not shown
  twenty near-duplicates.
* **Ambiguity** (a *penalty*) -- every answer in the request is unsure. That is
  irreducible uncertainty: the item is hard because the labels are arbitrary, not
  because the model lacks a feature. Pure uncertainty sampling marches the human
  straight into this region -- in the sentiment corpus, the neutral tier, where
  every model sits near chance and the ceiling is about 0.87 overall. Chase
  *reducible* uncertainty; down-weight the rest.

Selection is **stochastic** on purpose. The scores become a probability
distribution (a softmax with a uniform exploration floor), an item is sampled from
it, and the probability it was sampled with is recorded. That recorded propensity is
what lets the fit inverse-probability-weight the labels and undo the bias selection
introduces. A rule that always shows the top-scoring item gives it probability one
and every other item zero, and there would be nothing to invert.
"""
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from jev_flywheel.answers import AnswerCache
from jev_flywheel.head import contributions as head_contributions
from jev_flywheel.items import Item, normalize_label
from jev_flywheel.sampling import selection_distribution
from jev_flywheel.scorecard import Score
from jev_flywheel.scoring import ScoreResult, predict

POLICY_NAME = "selection-v1"


@dataclass(frozen=True)
class SelectionPolicy:
    """Frozen and named, so a recorded selection stays interpretable later."""

    name: str = POLICY_NAME
    uncertainty: float = 1.0
    disagreement: float = 1.0
    conflict: float = 1.0
    novelty: float = 0.5
    ambiguity: float = 1.0       # subtracted
    temperature: float = 0.35
    explore: float = 0.10


@dataclass
class Candidate:
    item_id: str
    result: ScoreResult
    vector: Dict[str, float]
    components: Dict[str, float]
    score: float = 0.0


@dataclass
class Selection:
    """The chosen item and the probability it was chosen with."""

    candidate: Candidate
    propensity: float
    policy: str
    pool_size: int

    def record(self) -> Dict[str, Any]:
        """What is stored beside the label, so the fit can weight it and a reader can see why."""
        return {
            "propensity": self.propensity,
            "selection_policy": self.policy,
            "selection_score": round(self.candidate.score, 4),
            "selection_components": {k: round(v, 4) for k, v in self.candidate.components.items()},
            "pool_size": self.pool_size,
        }


def binary_entropy(p: float) -> float:
    """Entropy of a two-outcome distribution, scaled to 0..1."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _distribution_entropy(probabilities: Sequence[float]) -> float:
    total = sum(probabilities)
    if len(probabilities) < 2 or total <= 0:
        return 0.0
    h = -sum((p / total) * math.log(p / total) for p in probabilities if p > 0)
    return h / math.log(len(probabilities))


def answer_entropy(answers: Mapping[str, Mapping[str, Any]]) -> float:
    """Mean normalized entropy across every answer in the request.

    High when the whole request is unsure. That is the signature of an item that is
    hard because its label is arbitrary, which more labels will not fix.
    """
    entropies: List[float] = []
    for answer in answers.values():
        if answer.get("type") == "noul" or "noul" in answer:
            entropies.append(binary_entropy(float(answer["noul"])))
        elif answer.get("probabilities"):
            entropies.append(_distribution_entropy(
                [float(v) for v in answer["probabilities"].values()]))
    return sum(entropies) / len(entropies) if entropies else 0.0


def evidence_conflict(contribution: Mapping[str, float]) -> float:
    """The share of evidence that argues against the decision, in 0..0.5.

    Zero when every feature agrees. Near 0.5 when for and against are evenly
    matched, which is the interesting case: the head made a call while its own
    inputs disagreed about it.
    """
    positive = sum(v for v in contribution.values() if v > 0)
    negative = -sum(v for v in contribution.values() if v < 0)
    total = positive + negative
    return min(positive, negative) / total if total > 0 else 0.0


def _novelty(vectors: List[Dict[str, float]], labeled: List[Dict[str, float]],
             features: Sequence[str]) -> List[float]:
    """Distance to the nearest labeled item, squashed to 0..1 and standardized per feature."""
    import numpy as np

    if not vectors:
        return []
    if not labeled or not features:
        return [0.5] * len(vectors)

    def matrix(rows):
        return np.array([[row.get(name, 0.0) for name in features] for row in rows], dtype=float)

    pool, seen = matrix(vectors), matrix(labeled)
    scale = pool.std(axis=0)
    scale[scale == 0] = 1.0
    pool, seen = pool / scale, seen / scale
    nearest = np.empty(len(pool))
    for start in range(0, len(pool), 512):        # chunked so a big pool stays in memory
        block = pool[start:start + 512]
        distances = np.sqrt(((block[:, None, :] - seen[None, :, :]) ** 2).sum(axis=2))
        nearest[start:start + 512] = distances.min(axis=1)
    middle = float(np.median(nearest)) or 1.0
    return (nearest / (nearest + middle)).tolist()


def build_candidates(
    score: Score,
    questions: Mapping[str, Mapping[str, Any]],
    cache: AnswerCache,
    items: Iterable[Item],
    labeled_ids: Set[str],
    *,
    policy: SelectionPolicy = SelectionPolicy(),
) -> List[Candidate]:
    """Score every unlabeled item for how much a label on it would teach."""
    pool = [item for item in items if item.id not in labeled_ids]
    answers_by_item = cache.bulk_partial_answers(
        [item.id for item in pool] + sorted(labeled_ids), questions)
    features = list(score.decision.features) if score.decision else []

    candidates: List[Candidate] = []
    for item in pool:
        answers = answers_by_item[item.id]
        if score.question_name not in answers and score.decision is None:
            continue                                  # nothing to score with
        vector = score.feature_vector(answers)
        result = predict(score, answers, vector)
        detail = result.metadata.get("decision") or {}
        jev = result.metadata.get("jev") or {}
        disagree = 0.0
        if jev.get("value") is not None:
            disagree = float(normalize_label(jev["value"]) != normalize_label(result.value))
        conflict = 0.0
        probabilities = detail.get("probabilities")
        if probabilities and score.decision is not None:
            ranked = sorted(probabilities, key=probabilities.get, reverse=True)
            conflict = evidence_conflict(
                head_contributions(vector, score.decision.head(), ranked[0], ranked[1]))
        candidates.append(Candidate(
            item_id=item.id, result=result, vector=vector,
            components={
                "uncertainty": binary_entropy(result.confidence or 0.5),
                "disagreement": disagree,
                "conflict": conflict,
                "ambiguity": answer_entropy(answers),
            }))

    labeled_vectors = [score.feature_vector(answers_by_item[i]) for i in sorted(labeled_ids)
                       if answers_by_item.get(i)]
    for candidate, novelty in zip(candidates, _novelty(
            [c.vector for c in candidates], labeled_vectors, features)):
        candidate.components["novelty"] = novelty
        c = candidate.components
        candidate.score = (policy.uncertainty * c["uncertainty"]
                           + policy.disagreement * c["disagreement"]
                           + policy.conflict * c["conflict"]
                           + policy.novelty * c["novelty"]
                           - policy.ambiguity * c["ambiguity"])
    return candidates


def choose(candidates: Sequence[Candidate], rng: random.Random,
           policy: SelectionPolicy = SelectionPolicy()) -> Selection:
    """Sample one candidate, and record the probability it was sampled with."""
    if not candidates:
        raise ValueError("there are no unlabeled items left to ask about")
    probabilities = selection_distribution(
        [c.score for c in candidates], temperature=policy.temperature, explore=policy.explore)
    index = rng.choices(range(len(candidates)), weights=probabilities, k=1)[0]
    return Selection(candidates[index], probabilities[index], policy.name, len(candidates))
