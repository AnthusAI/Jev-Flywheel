"""Talking to Jev: one request per item, carrying every question.

The invariant this module exists to protect: **one item costs one request**. Jev
charges for input tokens, which are dominated by the item's text, so the text is
paid for once and the questions ride along nearly free. In the sentiment corpus,
eight questions averaged about 501 input tokens per item against about 335 for
two. That ratio is what makes decomposing a score into elements affordable, and
the same decomposition against ordinary LLM prompts would multiply cost by the
number of elements instead.

So every code path here funnels into a single ``system_one`` call per item, and
concurrent callers asking about the same item await the same in-flight request
rather than issuing a second one.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

DEFAULT_MAX_CACHE_ENTRIES = 1024


def _default_client():
    """Build the real SDK client.

    Imported lazily and only here, so the rest of the project imports, and its
    specs run, without the SDK installed.
    """
    try:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    except ImportError as error:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "Calling Jev requires the 'typesafe-sdk' package (pip install typesafe-sdk)."
        ) from error
    # The key and base URL come from TYPESAFE_API_KEY / TYPESAFE_BASE_URL. They
    # are never passed in as arguments, so they cannot end up in a log line.
    return AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0))


def _as_dict(value: Any) -> dict:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def fingerprint(payload: Any) -> str:
    """A stable content hash, used for both questions and question sets."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def normalize_answer(answer: Mapping[str, Any]) -> Dict[str, Any]:
    """Make one answer safe to cache and to read back.

    Score answers key their probabilities and legend by integer level, which JSON
    cannot represent as an object key. Normalizing to strings on the way in means
    a cached answer and a live one behave identically -- the alternative is a
    feature extractor that works on fixtures and silently returns zeros in
    production.
    """
    out = dict(answer)
    for key in ("probabilities", "legend"):
        value = out.get(key)
        if isinstance(value, Mapping):
            out[key] = {str(k): v for k, v in value.items()}
    return out


@dataclass(frozen=True)
class JevAnswers:
    """Everything one request returned. The only thing that crosses the boundary."""

    answers: Dict[str, dict]
    model: Optional[str] = None
    usage: Optional[dict] = None


@dataclass
class JevSession:
    """Holds the question set for a scorecard and issues the requests.

    ``questions`` is the whole card's question set: registering the same name
    twice with an identical body is allowed, which is how several scores share an
    element, and registering it with a different body raises, which turns an
    authoring mistake into a loud failure instead of a silent overwrite.
    """

    client_factory: Any = _default_client
    max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES
    questions: Dict[str, dict] = field(default_factory=dict)
    requests_sent: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        self._client = None
        self._cache: "OrderedDict[str, asyncio.Future]" = OrderedDict()
        self._owners: Dict[str, str] = {}
        self._fingerprint: Optional[str] = None

    def register_question(self, name: str, question: Mapping[str, Any],
                          owner: Optional[str] = None) -> None:
        existing = self.questions.get(name)
        body = dict(question)
        if existing is not None and existing != body:
            raise ValueError(
                f"question {name!r} is already registered by {self._owners.get(name)!r} "
                f"with a different definition; {owner!r} cannot redefine it")
        self.questions[name] = body
        self._owners.setdefault(name, owner or "unknown")
        self._fingerprint = None  # the set changed, so the memo is stale

    def register_scorecard(self, scorecard) -> None:
        """Register every question on a whole scorecard."""
        for score in scorecard.scores:
            for name, question in score.questions().items():
                self.register_question(name, question, owner=score.name)

    @property
    def question_set_fingerprint(self) -> str:
        """Identifies the whole question set. Memoized; invalidated on register."""
        if self._fingerprint is None:
            self._fingerprint = fingerprint(self.questions)
        return self._fingerprint

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.question_set_fingerprint}\x00{text}".encode("utf-8")).hexdigest()

    async def answer_all(self, text: str) -> JevAnswers:
        """Answer every registered question about one item, in one request.

        The cache holds in-flight futures rather than results, so concurrent
        callers asking about the same item share one request instead of racing.
        A failure is not cached, so a retry is free to try again.
        """
        key = self._cache_key(text)
        future = self._cache.get(key)
        if future is not None:
            self._cache.move_to_end(key)
            return await future

        future = asyncio.get_running_loop().create_future()
        self._cache[key] = future
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        try:
            future.set_result(await self.ask(text, dict(self.questions)))
        except BaseException as error:
            self._cache.pop(key, None)
            future.set_exception(error)
            future.exception()  # mark retrieved so asyncio does not warn
            raise
        return await future

    async def ask(self, text: str, questions: Mapping[str, dict]) -> JevAnswers:
        """One request for an explicit question set.

        Used directly when topping up the answer cache for newly proposed
        elements, where re-asking the questions already on file would throw away
        the whole point of caching per question.
        """
        if not questions:
            return JevAnswers(answers={})
        if self._client is None:
            self._client = self.client_factory()
        response = await self._client.system_one(state={"text": text}, questions=dict(questions))
        self.requests_sent += 1
        usage = _as_dict(response.usage) if getattr(response, "usage", None) else None
        if usage:
            # Token counts are optional in the SDK's model, and a present-but-None
            # value would raise on +=.
            self.input_tokens += usage.get("input_tokens") or 0
            self.output_tokens += usage.get("output_tokens") or 0
        answers = {
            name: normalize_answer(_as_dict(answer))
            for name, answer in (response.answers or {}).items()
        }
        return JevAnswers(answers=answers, model=getattr(response, "model", None), usage=usage)
