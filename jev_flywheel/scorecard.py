"""The whole-scorecard configuration: one YAML file, loaded and validated.

One file for the whole scorecard, because that mirrors Jev's architecture: a
single request carries every question about an item, and the cost is dominated by
the item's text rather than by the number of questions. Splitting the
configuration per score would hide the thing that makes this design work.

This file is also the artifact the steering procedure evolves. Its version
lineage is the flywheel's history, so it has to round-trip cleanly and validate
strictly: a malformed candidate must fail at load time, with a message naming
what is wrong, rather than halfway through a fit.

Validation is hand-written rather than delegated to a schema library so the error
messages can explain *why* a rule exists. This is a project people read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from jev_flywheel.features import HOLISTIC, available_terms, extract_terms, split_feature
from jev_flywheel.head import validate_head

QUESTION_TYPES = ("noul", "choice", "score")
RESERVED_ELEMENT_KEYS = frozenset({"self", "shared"})
ELEMENT_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_CLIP = 0.01


class ConfigError(ValueError):
    """A scorecard that cannot be loaded. The message names the offending key."""


def slug(text: str) -> str:
    """A deterministic wire-safe name from a score's key or name.

    Score names contain spaces and dots, and keys are often absent, so the wire
    name is derived rather than taken verbatim.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _require_criteria(question_type: str, criteria: Any, where: str) -> None:
    """Choice and score questions must carry non-empty criteria.

    The SDK only checks that the key is present, so a None slips through to the
    API and fails server-side with a far less useful message.
    """
    if question_type in ("choice", "score"):
        if not criteria:
            raise ConfigError(
                f"{where}: a {question_type} question needs non-empty criteria; "
                "without them the request fails at the API instead of here")
        if question_type == "score" and len(_as_options(criteria)) < 2:
            raise ConfigError(f"{where}: a score question needs at least two levels")
        raw_options = list(criteria) if not isinstance(criteria, Mapping) else list(criteria.keys())
        _reject_yaml_booleans(raw_options, where, "criteria")


def _as_options(criteria: Any) -> List[str]:
    if isinstance(criteria, Mapping):
        return [str(k) for k in criteria]
    return [str(v) for v in (criteria or [])]


def _reject_yaml_booleans(values: Iterable[Any], where: str, what: str) -> None:
    """Refuse label strings that YAML turned into booleans.

    YAML 1.1 reads unquoted ``Yes``, ``No``, ``On`` and ``Off`` as booleans, so
    ``classes: [Yes, No]`` arrives as ``[True, False]`` and every label
    comparison then fails against the string "yes". Nothing raises; the metrics
    just go to zero. Catching it here, with an actionable message, is cheap --
    and it matters because the steering agent writes this file.
    """
    for value in values:
        if isinstance(value, bool):
            raise ConfigError(
                f"{where}: {what} contains the boolean {value!r}, which means a bare "
                "Yes/No/On/Off was written unquoted. Quote it, as in "
                '\'classes: ["Yes", "No"]\'')


@dataclass
class ElementSpec:
    """One sub-question answered in the same Jev request as the score's own.

    An element is evidence, not a verdict. It is deliberately cheap: adding one
    costs a few tokens in a request that was being sent anyway.
    """

    key: str
    question_type: str
    instructions: Any = None
    criteria: Any = None

    def validate(self, where: str) -> None:
        if not ELEMENT_KEY.match(self.key or ""):
            raise ConfigError(
                f"{where}: element key {self.key!r} must match [A-Za-z0-9_-]+ and "
                "contain no dots, because feature names split on dots")
        if self.key in RESERVED_ELEMENT_KEYS:
            raise ConfigError(f"{where}: element key {self.key!r} is reserved")
        if self.question_type not in QUESTION_TYPES:
            raise ConfigError(
                f"{where}: element {self.key!r} has question_type {self.question_type!r}; "
                f"expected one of {QUESTION_TYPES}")
        _require_criteria(self.question_type, self.criteria, f"{where}: element {self.key!r}")

    def question(self) -> Dict[str, Any]:
        """The wire form of this question."""
        return build_question(
            question_type=self.question_type, instructions=self.instructions,
            criteria=self.criteria)

    def to_config(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"key": self.key, "question_type": self.question_type}
        if self.instructions is not None:
            out["instructions"] = self.instructions
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


@dataclass
class DecisionSpec:
    """The model that turns features into the final value and a confidence.

    ``weights`` are keyed by feature name, never by position, so adding an
    element never scrambles the existing weights and a feature that is absent at
    serving time simply counts as zero.
    """

    model: str
    classes: List[str]
    features: List[str]
    weights: Dict[str, Any] = field(default_factory=dict)
    positive_class: Optional[str] = None
    threshold: float = 0.0
    clip: float = DEFAULT_CLIP
    abstain_band: float = 0.0
    calibration: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None

    def head(self) -> Dict[str, Any]:
        """The plain dict the serving code evaluates."""
        return {
            "model": self.model,
            "classes": list(self.classes),
            "weights": self.weights,
            "positive_class": self.positive_class,
            "threshold": self.threshold,
            "abstain_band": self.abstain_band,
        }

    def to_config(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model": self.model,
            "classes": list(self.classes),
            "features": list(self.features),
            "parameters": {"weights": self.weights},
        }
        if self.positive_class is not None:
            out["positive_class"] = self.positive_class
        if self.threshold:
            out["threshold"] = self.threshold
        if self.clip != DEFAULT_CLIP:
            out["clip"] = self.clip
        if self.abstain_band:
            out["abstain_band"] = self.abstain_band
        if self.calibration:
            out["calibration"] = self.calibration
        if self.provenance:
            out["provenance"] = self.provenance
        return out

    @classmethod
    def from_config(cls, raw: Mapping[str, Any], where: str) -> "DecisionSpec":
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{where}: decision must be a mapping")
        for required in ("model", "classes", "features"):
            if required not in raw:
                raise ConfigError(f"{where}: decision is missing {required!r}")
        parameters = raw.get("parameters") or {}
        # Check before str() coercion, which would turn True into "True".
        _reject_yaml_booleans(raw["classes"], where, "decision classes")
        if raw.get("positive_class") is not None:
            _reject_yaml_booleans([raw["positive_class"]], where, "positive_class")
        return cls(
            model=str(raw["model"]),
            classes=[str(c) for c in raw["classes"]],
            features=_feature_names(raw["features"], where),
            weights=dict(parameters.get("weights") or {}),
            positive_class=raw.get("positive_class"),
            threshold=float(raw.get("threshold", 0.0)),
            clip=float(raw.get("clip", DEFAULT_CLIP)),
            abstain_band=float(raw.get("abstain_band", 0.0)),
            calibration=raw.get("calibration"),
            provenance=raw.get("provenance"),
        )


def _feature_names(raw: Iterable[Any], where: str) -> List[str]:
    """Feature entries may be bare strings or ``{name: ...}`` mappings."""
    names: List[str] = []
    for entry in raw or []:
        if isinstance(entry, Mapping):
            if "name" not in entry:
                raise ConfigError(f"{where}: a feature mapping needs a 'name'")
            names.append(str(entry["name"]))
        else:
            names.append(str(entry))
    return names


def build_question(*, question_type: str, instructions: Any = None,
                   criteria: Any = None) -> Dict[str, Any]:
    """The wire form of one question.

    ``criteria`` is omitted entirely when absent rather than sent as null,
    because the API rejects a null where it expects options.
    """
    question: Dict[str, Any] = {"type": question_type}
    if instructions is not None:
        question["instructions"] = instructions
    if criteria:
        question["criteria"] = criteria
    return question


@dataclass
class Score:
    """One score: its own holistic question, its elements, and its decision."""

    name: str
    key: Optional[str] = None
    question_type: Optional[str] = None
    instructions: Any = None
    criteria: Any = None
    elements: List[ElementSpec] = field(default_factory=list)
    shared_elements: List[ElementSpec] = field(default_factory=list)
    decision: Optional[DecisionSpec] = None

    @property
    def slug(self) -> str:
        return slug(self.key or self.name)

    @property
    def question_name(self) -> str:
        """The wire name of the score's own holistic question."""
        return self.name

    def validate(self) -> None:
        where = f"score {self.name!r}"
        if not self.name:
            raise ConfigError("every score needs a name")
        if self.question_type is None and self.decision is None:
            raise ConfigError(
                f"{where}: needs a question_type, a decision, or both; "
                "otherwise it has no way to produce a value")
        if self.question_type is not None:
            if self.question_type not in QUESTION_TYPES:
                raise ConfigError(
                    f"{where}: question_type {self.question_type!r}; expected one of "
                    f"{QUESTION_TYPES}")
            _require_criteria(self.question_type, self.criteria, where)

        # Own and shared keys live in separate namespaces, so scope the check
        # rather than testing list membership: an own element that happens to
        # equal a shared one must not be mistaken for it.
        for scope, specs in (("own", self.elements), ("shared", self.shared_elements)):
            seen = set()
            for spec in specs:
                spec.validate(where)
                if spec.key in seen:
                    raise ConfigError(f"{where}: duplicate {scope} element key {spec.key!r}")
                seen.add(spec.key)

        if self.decision is not None:
            self._validate_decision(where)

    def refs(self) -> Dict[str, Tuple[str, Any]]:
        """Every feature reference this score can resolve, to ``(type, criteria)``."""
        table: Dict[str, Tuple[str, Any]] = {}
        if self.question_type is not None:
            table[HOLISTIC] = (self.question_type, self.criteria)
        for spec in self.elements:
            table[spec.key] = (spec.question_type, spec.criteria)
        for spec in self.shared_elements:
            table[f"shared.{spec.key}"] = (spec.question_type, spec.criteria)
        return table

    def _validate_decision(self, where: str) -> None:
        decision = self.decision
        assert decision is not None
        table = self.refs()
        for name in decision.features:
            ref, term = split_feature(name)
            if ref not in table:
                raise ConfigError(
                    f"{where}: decision feature {name!r} refers to {ref!r}, which is not a "
                    f"declared element; declared: {sorted(table)}")
            question_type, criteria = table[ref]
            if term not in available_terms(question_type, criteria):
                raise ConfigError(
                    f"{where}: decision feature {name!r} asks for {term!r}, which a "
                    f"{question_type} question cannot produce")
        problems = validate_head(decision.head(), features=decision.features)
        if problems:
            raise ConfigError(f"{where}: " + "; ".join(problems))

    def element_questions(self) -> List[Tuple[str, str, ElementSpec]]:
        """``(feature ref, wire name, spec)`` for every element.

        Own elements are namespaced by score on the wire but referenced bare in
        features, which keeps a decision block portable between scores. Shared
        elements use the same name in both places so that several scores
        deliberately collide on one registration.
        """
        own = [(s.key, f"{self.slug}.{s.key}", s) for s in self.elements]
        shared = [(f"shared.{s.key}", f"shared.{s.key}", s) for s in self.shared_elements]
        return own + shared

    def questions(self) -> Dict[str, Dict[str, Any]]:
        """Every question this score contributes to the one shared request."""
        out: Dict[str, Dict[str, Any]] = {}
        if self.question_type is not None:
            out[self.question_name] = build_question(
                question_type=self.question_type, instructions=self.instructions,
                criteria=self.criteria)
        for _, wire_name, spec in self.element_questions():
            out[wire_name] = spec.question()
        return out

    def feature_vector(self, answers: Mapping[str, Any]) -> Dict[str, float]:
        """The declared features, from a mapping of wire question name to answer.

        A reference with no answer is omitted rather than zero-filled, so the
        serving path can report coverage and the fitting path can refuse.
        """
        if self.decision is None:
            return {}
        clip = self.decision.clip
        sources: Dict[str, Tuple[str, str, Any]] = {}
        if self.question_type is not None:
            sources[HOLISTIC] = (self.question_name, self.question_type, self.criteria)
        for ref, wire_name, spec in self.element_questions():
            sources[ref] = (wire_name, spec.question_type, spec.criteria)

        cache: Dict[str, Dict[str, float]] = {}
        vector: Dict[str, float] = {}
        for name in self.decision.features:
            ref, term = split_feature(name)
            if ref not in cache:
                wire_name, question_type, criteria = sources[ref]
                answer = answers.get(wire_name)
                cache[ref] = ({} if answer is None
                              else extract_terms(answer, question_type, criteria, clip))
            if term in cache[ref]:
                vector[name] = cache[ref][term]
        return vector

    def to_config(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.key:
            out["key"] = self.key
        if self.question_type is not None:
            out["question_type"] = self.question_type
        if self.instructions is not None:
            out["instructions"] = self.instructions
        if self.criteria is not None:
            out["criteria"] = self.criteria
        if self.elements:
            out["elements"] = [s.to_config() for s in self.elements]
        if self.shared_elements:
            out["shared_elements"] = [s.to_config() for s in self.shared_elements]
        if self.decision is not None:
            out["decision"] = self.decision.to_config()
        return out

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "Score":
        if not isinstance(raw, Mapping):
            raise ConfigError("each entry under 'scores' must be a mapping")
        name = str(raw.get("name") or "")
        where = f"score {name!r}"
        decision = raw.get("decision")
        return cls(
            name=name,
            key=raw.get("key"),
            question_type=raw.get("question_type"),
            instructions=raw.get("instructions"),
            criteria=raw.get("criteria"),
            elements=[_element(e, where) for e in raw.get("elements") or []],
            shared_elements=[_element(e, where) for e in raw.get("shared_elements") or []],
            decision=DecisionSpec.from_config(decision, where) if decision else None,
        )


def _element(raw: Mapping[str, Any], where: str) -> ElementSpec:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: each element must be a mapping")
    if "key" not in raw:
        raise ConfigError(f"{where}: every element needs a key")
    return ElementSpec(
        key=str(raw["key"]),
        question_type=str(raw.get("question_type") or ""),
        instructions=raw.get("instructions"),
        criteria=raw.get("criteria"),
    )


@dataclass
class Scorecard:
    """Every score on the card, answered by one Jev request per item."""

    name: str
    scores: List[Score] = field(default_factory=list)
    version: int = 1

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("a scorecard needs a name")
        if not self.scores:
            raise ConfigError("a scorecard needs at least one score")
        names = [s.name for s in self.scores]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ConfigError(f"duplicate score names: {sorted(duplicates)}")
        for score in self.scores:
            score.validate()
        self.questions()  # surfaces cross-score wire-name conflicts

    def score(self, name: str) -> Score:
        for candidate in self.scores:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def questions(self) -> Dict[str, Dict[str, Any]]:
        """The single question set for one Jev request covering the whole card.

        Two scores may register the same shared element, which is the point, but
        only if the definitions agree. A disagreement is an authoring error and
        must fail loudly rather than let one silently overwrite the other.
        """
        out: Dict[str, Dict[str, Any]] = {}
        owners: Dict[str, str] = {}
        for score in self.scores:
            for wire_name, question in score.questions().items():
                if wire_name in out and out[wire_name] != question:
                    raise ConfigError(
                        f"question {wire_name!r} is defined differently by "
                        f"{owners[wire_name]!r} and {score.name!r}")
                out[wire_name] = question
                owners.setdefault(wire_name, score.name)
        return out

    def to_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "scores": [s.to_config() for s in self.scores],
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_config(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "Scorecard":
        if not isinstance(raw, Mapping):
            raise ConfigError("a scorecard must be a mapping")
        if "scores" not in raw:
            raise ConfigError("a scorecard needs a 'scores' list")
        card = cls(
            name=str(raw.get("name") or ""),
            scores=[Score.from_config(s) for s in raw["scores"]],
            version=int(raw.get("version", 1)),
        )
        card.validate()
        return card

    @classmethod
    def from_yaml(cls, text: str) -> "Scorecard":
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ConfigError(f"not valid YAML: {error}") from error
        return cls.from_config(raw)
