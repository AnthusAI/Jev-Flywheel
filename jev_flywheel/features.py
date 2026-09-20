"""Named numeric features from Jev answers.

Each feature is a deterministic function of the configured criteria and one
answer. There is no fitted preprocessing anywhere, so all data dependence lives
in the weights, the weights stay keyed by feature name rather than position, and
a human can author a head by hand.

Features are ``<element>.<term>``. Terms by question type:

    noul    logit_p, p, is_yes
    choice  clr.<option>, chosen.<option>, top_p, entropy, confidence
    score   score_norm, expected_level, clr.<level index>, confidence

Three decisions that are easy to get wrong:

1. **Centered log-ratio, not per-option logits.** Choice and score probabilities
   are compositional: they sum to one, so K per-option logits are collinear.
   Regularization then splits weight arbitrarily among them and two refits on
   identical data give different, non-comparable weights -- which makes the
   feature importances that drive the optimizer noise. The CLR sums to zero,
   needs no reference option, and is stable.
2. **Clip hard.** Jev's tails are not calibrated and answers flip about 1% of
   the time between identical runs. Unclipped, one flipped answer from 0.995 to
   0.005 is a swing of more than ten units in log-odds space, enough for a
   single noisy element to dominate the decision.
3. **Probability keys may be integers.** Jev's score answers key probabilities
   by level number, and the SDK models that as ``dict[int, float]``. Look up
   both spellings: a JSONL round trip stringifies the keys, so code that only
   handles strings works on cached data and silently returns all zeros on the
   live path.
"""
import math
from typing import Any, Dict, List, Mapping, Set, Tuple

HOLISTIC = "self.holistic"


def _clip(p: float, eps: float) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def _logit(p: float, eps: float) -> float:
    p = _clip(p, eps)
    return math.log(p / (1.0 - p))


def _options(criteria: Any) -> List[str]:
    """The configured options, in configured order.

    Order comes from the criteria and never from the answer, because an answer
    may omit options and a contract that depends on model output is not a
    contract.
    """
    if isinstance(criteria, Mapping):
        return [str(key) for key in criteria]
    return [str(item) for item in (criteria or [])]


def _probability(given: Mapping[Any, Any], key: Any) -> float:
    """Read one probability, tolerating integer or string keys.

    See decision 3 in the module docstring: this is the difference between
    working on live answers and silently producing zeros.
    """
    if key in given:
        return float(given[key])
    text = str(key)
    if text in given:
        return float(given[text])
    return 0.0


def _clr(probabilities: List[float], eps: float) -> List[float]:
    logs = [math.log(max(p, eps)) for p in probabilities]
    mean = sum(logs) / len(logs)
    return [value - mean for value in logs]


def _normalized_entropy(probabilities: List[float]) -> float:
    total = sum(probabilities)
    if len(probabilities) < 2 or total <= 0:
        return 0.0
    entropy = -sum((p / total) * math.log(p / total) for p in probabilities if p > 0)
    return entropy / math.log(len(probabilities))


def split_feature(name: str) -> Tuple[str, str]:
    """Split ``<element>.<term>`` into ``(element ref, term)``.

    Element keys never contain dots, which makes this exact: ``self.holistic.*``
    refers to the score's own question, ``shared.<key>.*`` to a shared element,
    and anything else to one of the score's own elements.
    """
    if name.startswith(HOLISTIC + "."):
        return HOLISTIC, name[len(HOLISTIC) + 1:]
    if name.startswith("shared."):
        _, key, term = (name.split(".", 2) + ["", ""])[:3]
        return f"shared.{key}", term
    ref, _, term = name.partition(".")
    return ref, term


def available_terms(question_type: str, criteria: Any) -> Set[str]:
    """Every term ``extract_terms`` can produce for a question.

    Used to reject a scorecard at load time when a decision references a term
    its element cannot produce, rather than discovering it while serving.
    """
    if question_type == "noul":
        return {"logit_p", "p", "is_yes"}
    if question_type == "choice":
        options = _options(criteria)
        return ({"top_p", "entropy", "confidence"}
                | {f"clr.{o}" for o in options} | {f"chosen.{o}" for o in options})
    if question_type == "score":
        levels = range(len(_options(criteria)))
        return {"score_norm", "expected_level", "confidence"} | {f"clr.{i}" for i in levels}
    raise ValueError(f"Unsupported question type: {question_type!r}")


def extract_terms(
    answer: Mapping[str, Any], question_type: str, criteria: Any, eps: float
) -> Dict[str, float]:
    """Turn one Jev answer into its named terms."""
    if question_type == "noul":
        p = float(answer["noul"])
        return {"logit_p": _logit(p, eps), "p": p, "is_yes": 1.0 if p >= 0.5 else 0.0}

    terms: Dict[str, float] = {}
    if question_type == "choice":
        options = _options(criteria)
        given = answer.get("probabilities") or {}
        probabilities = [_probability(given, option) for option in options]
        for option, value in zip(options, _clr(probabilities, eps)):
            terms[f"clr.{option}"] = value
        for option in options:
            terms[f"chosen.{option}"] = 1.0 if answer.get("choice") == option else 0.0
        terms["top_p"] = max(probabilities)
        terms["entropy"] = _normalized_entropy(probabilities)
    elif question_type == "score":
        levels = len(_options(criteria))
        legend = answer.get("legend")
        if legend is not None and len(legend) != levels:
            # A legend that disagrees with the configured criteria means the
            # rubric changed under us. Fail loudly; do not guess an alignment.
            raise ValueError(
                f"Jev legend has {len(legend)} levels but the configured criteria has {levels}")
        given = answer.get("probabilities") or {}
        probabilities = [_probability(given, i) for i in range(levels)]
        span = max(levels - 1, 1)
        total = sum(probabilities) or 1.0
        terms["score_norm"] = float(answer["score"]) / span
        terms["expected_level"] = sum(i * p for i, p in enumerate(probabilities)) / total / span
        for i, value in enumerate(_clr(probabilities, eps)):
            terms[f"clr.{i}"] = value
    else:
        raise ValueError(f"Unsupported question type: {question_type!r}")

    if answer.get("confidence") is not None:
        terms["confidence"] = float(answer["confidence"])
    return terms
