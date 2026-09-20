"""Feature: one item costs one Jev request.

The whole economic argument for elements rests on this, so it is spec'd from
several angles with a fake client that counts calls.
"""
import asyncio

import pytest

from jev_flywheel.jev import JevSession, normalize_answer
from jev_flywheel.scorecard import Scorecard


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=10):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def model_dump(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


class FakeResponse:
    def __init__(self, answers, usage=None, model="jev-test"):
        self.answers = answers
        self.usage = usage if usage is not None else FakeUsage()
        self.model = model


class FakeClient:
    """Records every request so specs can assert how many were sent."""

    def __init__(self, answers=None, usage=None, delay=0.0):
        self._answers = answers or {}
        self._usage = usage
        self._delay = delay
        self.calls = []

    async def system_one(self, *, state, questions):
        self.calls.append(dict(questions))
        if self._delay:
            await asyncio.sleep(self._delay)
        return FakeResponse(dict(self._answers), usage=self._usage)


CARD = """
name: Card
scores:
  - name: Sentiment
    key: sentiment
    question_type: choice
    instructions: "Overall sentiment?"
    criteria: {positive: null, negative: null}
    elements:
      - {key: praise, question_type: noul, instructions: "Praise?"}
  - name: Escalate
    key: escalate
    question_type: noul
    instructions: "Escalate?"
"""


def session_for(client, card_yaml=CARD):
    session = JevSession(client_factory=lambda: client)
    session.register_scorecard(Scorecard.from_yaml(card_yaml))
    return session


async def test_a_whole_scorecard_is_one_request_carrying_every_question():
    client = FakeClient()
    session = session_for(client)

    await session.answer_all("some text")

    assert len(client.calls) == 1
    assert set(client.calls[0]) == {"Sentiment", "sentiment.praise", "Escalate"}


async def test_asking_about_the_same_item_twice_sends_one_request():
    client = FakeClient()
    session = session_for(client)

    await session.answer_all("same text")
    await session.answer_all("same text")

    assert session.requests_sent == 1


async def test_concurrent_callers_await_the_same_in_flight_request():
    # The cache holds futures, not results, so simultaneous callers share one
    # request rather than racing to issue two.
    client = FakeClient(delay=0.01)
    session = session_for(client)

    await asyncio.gather(*(session.answer_all("same text") for _ in range(5)))

    assert len(client.calls) == 1


async def test_different_items_each_cost_one_request():
    client = FakeClient()
    session = session_for(client)

    await session.answer_all("first")
    await session.answer_all("second")

    assert session.requests_sent == 2


async def test_a_failure_is_not_cached_so_a_retry_can_succeed():
    class Flaky:
        def __init__(self):
            self.calls = 0

        async def system_one(self, *, state, questions):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return FakeResponse({"Escalate": {"type": "noul", "noul": 0.5}})

    client = Flaky()
    session = JevSession(client_factory=lambda: client)
    session.register_question("Escalate", {"type": "noul"})

    with pytest.raises(RuntimeError):
        await session.answer_all("text")
    result = await session.answer_all("text")

    assert client.calls == 2
    assert "Escalate" in result.answers


async def test_token_usage_accumulates_across_requests():
    client = FakeClient(usage=FakeUsage(input_tokens=501, output_tokens=7))
    session = session_for(client)

    await session.answer_all("first")
    await session.answer_all("second")

    assert session.input_tokens == 1002
    assert session.output_tokens == 14


async def test_missing_token_counts_do_not_raise():
    # The SDK models both counts as optional, and a present-but-None value would
    # raise on +=.
    client = FakeClient(usage=FakeUsage(input_tokens=None, output_tokens=None))
    session = session_for(client)

    await session.answer_all("text")

    assert session.input_tokens == 0


def test_registering_the_same_question_twice_with_one_body_is_allowed():
    # This is how two scores share an element: one registration, asked once.
    session = JevSession(client_factory=lambda: FakeClient())
    body = {"type": "noul", "instructions": "Was it transferred?"}

    session.register_question("shared.transferred", dict(body), owner="A")
    session.register_question("shared.transferred", dict(body), owner="B")

    assert list(session.questions) == ["shared.transferred"]


def test_registering_the_same_name_with_a_different_body_raises():
    session = JevSession(client_factory=lambda: FakeClient())
    session.register_question("shared.t", {"type": "noul", "instructions": "one"}, owner="A")

    with pytest.raises(ValueError, match="shared.t"):
        session.register_question("shared.t", {"type": "noul", "instructions": "two"}, owner="B")


def test_the_question_set_fingerprint_changes_when_a_question_is_added():
    session = JevSession(client_factory=lambda: FakeClient())
    session.register_question("a", {"type": "noul"})
    before = session.question_set_fingerprint

    session.register_question("b", {"type": "noul"})

    assert session.question_set_fingerprint != before


def test_the_fingerprint_is_stable_for_the_same_set_in_a_different_order():
    first = JevSession(client_factory=lambda: FakeClient())
    first.register_question("a", {"type": "noul"})
    first.register_question("b", {"type": "noul"})
    second = JevSession(client_factory=lambda: FakeClient())
    second.register_question("b", {"type": "noul"})
    second.register_question("a", {"type": "noul"})

    assert first.question_set_fingerprint == second.question_set_fingerprint


async def test_adding_an_element_invalidates_the_in_memory_answer_cache():
    # Correct for the runtime cache: the answers on hand were produced by a
    # different question set. The on-disk cache is keyed per question precisely
    # so that this does not force a full re-ask.
    client = FakeClient()
    session = session_for(client)
    await session.answer_all("text")

    session.register_question("sentiment.irony", {"type": "noul", "instructions": "Irony?"})
    await session.answer_all("text")

    assert session.requests_sent == 2


async def test_an_explicit_subset_can_be_asked_without_the_rest():
    # This is how the cache tops up newly proposed elements without paying to
    # re-ask everything already on file.
    client = FakeClient()
    session = session_for(client)

    await session.ask("text", {"sentiment.irony": {"type": "noul", "instructions": "Irony?"}})

    assert list(client.calls[0]) == ["sentiment.irony"]


async def test_asking_nothing_sends_no_request():
    client = FakeClient()
    session = session_for(client)

    result = await session.ask("text", {})

    assert client.calls == []
    assert result.answers == {}


def test_integer_probability_keys_are_normalized_to_strings():
    # So a cached answer and a live one behave identically. Without this, the
    # feature extractor works on fixtures and returns zeros in production.
    answer = normalize_answer({"type": "score", "score": 2.0,
                               "probabilities": {0: 0.1, 1: 0.9},
                               "legend": {0: "none", 1: "mild"}})

    assert answer["probabilities"] == {"0": 0.1, "1": 0.9}
    assert answer["legend"] == {"0": "none", "1": "mild"}


async def test_answers_come_back_normalized_from_a_request():
    client = FakeClient(answers={"i": {"type": "score", "score": 1.0,
                                      "probabilities": {0: 0.5, 1: 0.5}}})
    session = session_for(client)

    result = await session.answer_all("text")

    assert result.answers["i"]["probabilities"] == {"0": 0.5, "1": 0.5}
    assert result.model == "jev-test"
