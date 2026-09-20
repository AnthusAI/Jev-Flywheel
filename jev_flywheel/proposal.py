"""The steering agent's proposals, and how they are applied.

The agent does not write YAML. It writes a small JSON *proposal of edits* -- add these
elements, retire those, reword the question on this one -- and this module applies it to
the current scorecard. That division is deliberate, and it buys several guarantees that
would otherwise be conventions asked of a language model:

* **The agent cannot touch weights, calibration or provenance.** The proposal format has
  nowhere to put them. Numbers come only from the deterministic fit, so a language model
  is never in the numeric path.
* **The change is exactly the change described.** No re-serialization drift, no
  accidentally reworded neighbour, no dropped key. The diff shown to the human is the
  diff that gets applied.
* **A proposal is one batch by construction.** Any change to the question set forces a
  fresh Jev pass over the labeled items, so ten changes must cost one pass, not ten.
  A proposal *is* the whole batch; there is no way to spell "and then also".
* **A malformed proposal fails here, with a message the agent can act on,** long before
  anything is fit or any Jev request is spent.

The format is parsed leniently -- models wrap JSON in prose and code fences -- and
validated strictly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from jev_flywheel.scorecard import ConfigError, ElementSpec, Scorecard

MAX_NEW_ELEMENTS = 4
HOLISTIC_KEYS = ("holistic", "self")


class ProposalError(ValueError):
    """The proposal cannot be used. The message is written to be fed back to the agent."""


@dataclass
class ElementAdd:
    key: str
    question_type: str
    instructions: str
    criteria: Any = None
    features: Optional[List[str]] = None       # default: one summary feature per element


@dataclass
class Reword:
    key: str
    instructions: str


@dataclass
class Proposal:
    root_cause: str = ""
    add: List[ElementAdd] = field(default_factory=list)
    retire: List[str] = field(default_factory=list)
    reword: List[Reword] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not (self.add or self.retire or self.reword)

    def summary(self) -> Dict[str, List[str]]:
        return {"added": [a.key for a in self.add], "retired": list(self.retire),
                "reworded": [r.key for r in self.reword]}


def default_features(key: str, question_type: str, criteria: Any) -> List[str]:
    """One summary feature per element, which is all the evidence usually supports.

    A yes/no element has one. A choice with K options has K centered log-ratios that sum
    to zero, so K - 1 carry all the information. A score has its expected level. Keeping
    the default small matters: the ladder's feature budget is a fraction of the effective
    sample size, and a wide element can spend it alone.
    """
    if question_type == "noul":
        return [f"{key}.logit_p"]
    if question_type == "choice":
        options = list(criteria.keys() if isinstance(criteria, Mapping) else criteria or [])
        return [f"{key}.clr.{option}" for option in options[:-1]] or [f"{key}.top_p"]
    if question_type == "score":
        return [f"{key}.expected_level"]
    raise ProposalError(f"unknown question_type {question_type!r} for element {key!r}; "
                        "use noul, choice or score")


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull one JSON object out of a model reply, tolerating prose and code fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise ProposalError(
        "the reply did not contain a JSON object. Reply with a single JSON object and "
        "nothing else.")


def _plain(value: Any) -> Any:
    """Turn Lua/Python proxy tables into plain dicts and lists."""
    if hasattr(value, "items") and not isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "values") and hasattr(value, "keys"):
        return [_plain(v) for v in value.values()]
    return value


def _as_criteria(value: Any, question_type: str, key: str) -> Any:
    if value in (None, "", [], {}):
        return None
    if question_type == "choice":
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, list):
            return {str(option): None for option in value}
    if question_type == "score":
        if isinstance(value, Mapping):
            return [str(k) for k in value]
        if isinstance(value, list):
            return [str(v) for v in value]
    if question_type == "noul":
        return None
    raise ProposalError(f"element {key!r}: criteria must be a list of options")


def parse_proposal(raw: Any) -> Proposal:
    """A model reply (text, or an already-parsed object) into a validated ``Proposal``."""
    data = _extract_json(raw) if isinstance(raw, str) else _plain(raw)
    if not isinstance(data, dict):
        raise ProposalError("the proposal must be a JSON object")

    def listed(name: str) -> List[Any]:
        value = data.get(name) or []
        if not isinstance(value, list):
            raise ProposalError(f"'{name}' must be a list")
        return value

    proposal = Proposal(root_cause=str(data.get("root_cause") or "").strip())
    for entry in listed("add_elements"):
        if not isinstance(entry, dict):
            raise ProposalError("each item in 'add_elements' must be an object")
        for required in ("key", "question_type", "instructions"):
            if not entry.get(required):
                raise ProposalError(f"an added element is missing '{required}'")
        kind = str(entry["question_type"])
        key = str(entry["key"])
        criteria = _as_criteria(entry.get("criteria"), kind, key)
        if kind in ("choice", "score") and not criteria:
            raise ProposalError(f"element {key!r} is a {kind} question, so it needs 'criteria'")
        features = entry.get("features")
        if features is not None and not (
                isinstance(features, list) and all(isinstance(f, str) for f in features)):
            raise ProposalError(f"element {key!r}: 'features' must be a list of strings")
        proposal.add.append(ElementAdd(key, kind, str(entry["instructions"]), criteria, features))
    proposal.retire = [str(k) for k in listed("retire_elements")]
    for entry in listed("reword_elements"):
        if not isinstance(entry, dict) or not entry.get("key") or not entry.get("instructions"):
            raise ProposalError("each item in 'reword_elements' needs 'key' and 'instructions'")
        proposal.reword.append(Reword(str(entry["key"]), str(entry["instructions"])))

    if len(proposal.add) > MAX_NEW_ELEMENTS:
        raise ProposalError(
            f"{len(proposal.add)} new elements is more than the {MAX_NEW_ELEMENTS} allowed in "
            "one proposal. Every new element spends feature budget that the labels may not "
            "support; keep the most valuable ones.")
    keys = [a.key for a in proposal.add] + proposal.retire + [r.key for r in proposal.reword]
    duplicated = {k for k in keys if keys.count(k) > 1}
    if duplicated:
        raise ProposalError(f"{sorted(duplicated)} appear more than once across the proposal; "
                            "each element may be added, retired or reworded, not several of these")
    return proposal


def apply_proposal(card: Scorecard, score_name: str, proposal: Proposal) -> Scorecard:
    """The scorecard that results from a proposal, with the head's numbers cleared.

    Weights, calibration and provenance are reset because they belong to the old feature
    set; the fit that follows sets them. Everything the proposal does not mention is
    carried over untouched.
    """
    candidate = Scorecard.from_config(card.to_config())
    score = candidate.score(score_name)
    if score.decision is None:
        raise ProposalError(f"score {score_name!r} has no decision block to extend")
    existing = {e.key: e for e in score.elements}

    for key in proposal.retire:
        if key in HOLISTIC_KEYS:
            raise ProposalError("the holistic question cannot be retired; it is a feature "
                                "like the others and the fit can shrink it to nothing")
        if key not in existing:
            raise ProposalError(f"cannot retire {key!r}: no such element. Current elements: "
                                f"{sorted(existing)}")
    for change in proposal.reword:
        if change.key in HOLISTIC_KEYS:
            score.instructions = change.instructions
        elif change.key in existing:
            existing[change.key].instructions = change.instructions
        else:
            raise ProposalError(f"cannot reword {change.key!r}: no such element. Current "
                                f"elements: {sorted(existing)}")
    for added in proposal.add:
        if added.key in existing:
            raise ProposalError(f"element {added.key!r} already exists; to change it, reword it")

    retired = set(proposal.retire)
    score.elements = [e for e in score.elements if e.key not in retired]
    features = [f for f in score.decision.features
                if f.split(".", 1)[0] not in retired]
    for added in proposal.add:
        score.elements.append(ElementSpec(
            key=added.key, question_type=added.question_type,
            instructions=added.instructions, criteria=added.criteria))
        for feature in added.features or default_features(
                added.key, added.question_type, added.criteria):
            if feature not in features:
                features.append(feature)

    decision = score.decision
    decision.features = features
    decision.model = "multinomial_logistic"
    decision.positive_class = None
    decision.threshold = 0.0
    decision.weights = {}
    decision.calibration = None
    decision.provenance = None
    try:
        candidate.validate()
    except ConfigError as error:
        raise ProposalError(f"the resulting scorecard is not valid: {error}") from error
    return candidate


def diff_summary(before: Scorecard, after: Scorecard, score_name: str) -> Dict[str, Any]:
    """What differs between two scorecard versions, for the human deciding whether to approve."""
    old, new = before.score(score_name), after.score(score_name)
    old_by = {e.key: e for e in old.elements}
    new_by = {e.key: e for e in new.elements}
    reworded = [k for k in new_by if k in old_by
                and (old_by[k].instructions != new_by[k].instructions
                     or old_by[k].criteria != new_by[k].criteria)]
    old_features = list(old.decision.features) if old.decision else []
    new_features = list(new.decision.features) if new.decision else []
    return {
        "added": [k for k in new_by if k not in old_by],
        "retired": [k for k in old_by if k not in new_by],
        "reworded": reworded,
        "holistic_reworded": old.instructions != new.instructions,
        "features_added": [f for f in new_features if f not in old_features],
        "features_removed": [f for f in old_features if f not in new_features],
    }
