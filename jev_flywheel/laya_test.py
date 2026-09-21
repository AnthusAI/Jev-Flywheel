"""Feature: Laya answers the same questions locally, and refuses what it would truncate.

Laya cuts the state, each option and the instructions to fit its window without saying so, so
the guard is the part that matters. It is spec'd against Laya's real tokenizer (a few hundred
kilobytes, fetched with the checkpoint), and skipped where laya-mlx is not installed.
"""
import asyncio

import pytest

from jev_flywheel.jev import JevSession
from jev_flywheel.laya import (
    LayaBudgetError, LayaClient, LayaError, to_laya_question, to_laya_state)

laya_mlx = pytest.importorskip("laya_mlx")


@pytest.fixture(scope="module")
def agent():
    return laya_mlx.load("aac6fef/laya-mlx")


def test_jevs_text_envelope_is_unwrapped_but_anything_else_is_left_alone():
    assert to_laya_state({"text": "hello"}) == "hello"
    assert to_laya_state({"text": "hello", "id": 3}) == {"text": "hello", "id": 3}
    assert to_laya_state("plain") == "plain"


def test_a_score_dict_becomes_the_ordered_list_laya_requires():
    q = {"type": "score", "instructions": "How strong?", "criteria": {"none": None, "mild": None}}
    assert to_laya_question(q)["criteria"] == ["none", "mild"]
    assert to_laya_question({**q, "criteria": ["none", "mild"]})["criteria"] == ["none", "mild"]


def test_a_choice_keeps_its_labels_and_a_noul_needs_none():
    q = {"type": "choice", "instructions": "?", "criteria": {"positive": None, "negative": None}}
    assert to_laya_question(q)["criteria"] == {"positive": None, "negative": None}
    assert "criteria" not in to_laya_question({"type": "noul", "instructions": "?"})


class FakeAgent:
    """Records what it was asked, so a spec can prove nothing reached a real checkpoint."""

    def __init__(self, real):
        self.tok, self.cfg, self._to_internal = real.tok, real.cfg, real._to_internal
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, dict(questions)))
        return {"model": "laya-rl-agent", "usage": {"input_tokens": 40, "output_tokens": 0},
                "answers": {name: {"type": q["type"], "noul": 0.7} for name, q in questions.items()}}


def test_it_answers_through_the_jev_session_it_replaces(agent):
    fake = FakeAgent(agent)
    session = JevSession(client_factory=lambda: LayaClient(agent=fake))
    session.register_question("praise", {"type": "noul", "instructions": "Praise?"})
    session.register_question("anger", {"type": "noul", "instructions": "Anger?"})

    answers = asyncio.run(session.answer_all("What a great game."))

    assert set(answers.answers) == {"praise", "anger"}
    assert session.requests_sent == 1                        # still one request per item
    assert answers.model.startswith("laya:")
    assert fake.calls[0][0] == "What a great game."          # unwrapped, not {"text": ...}


def test_the_state_is_refused_rather_than_silently_truncated(agent):
    fake = FakeAgent(agent)
    client = LayaClient(agent=fake)
    too_long = "the quick brown fox jumps over the lazy dog. " * 120

    with pytest.raises(LayaBudgetError, match=r"state is \d+ tokens but only \d+ are available"):
        asyncio.run(client.system_one(
            state={"text": too_long}, questions={"q": {"type": "noul", "instructions": "?"}}))

    assert fake.calls == []                                   # it never reached the model


def test_the_guard_names_how_much_would_have_been_dropped(agent):
    client = LayaClient(agent=FakeAgent(agent))
    with pytest.raises(LayaBudgetError) as raised:
        asyncio.run(client.system_one(
            state={"text": "word " * 600}, questions={"q": {"type": "noul", "instructions": "?"}}))
    assert "silently drop the last" in str(raised.value)


def test_an_item_the_size_of_this_corpus_fits_with_room_to_spare(agent):
    # The longest item in the sentiment corpus is 349 characters.
    client = LayaClient(agent=FakeAgent(agent))
    questions = {f"q{i}": {"type": "noul", "instructions": "Does the text express praise?"}
                 for i in range(12)}
    asyncio.run(client.system_one(state={"text": "x " * 175}, questions=questions))


def test_an_option_past_the_48_token_limit_is_refused(agent):
    client = LayaClient(agent=FakeAgent(agent))
    long_option = " ".join(["sentiment"] * 80)
    q = {"type": "choice", "instructions": "?",
         "criteria": {"positive": long_option, "negative": None}}
    with pytest.raises(LayaBudgetError, match="keeps 48 and drops the rest"):
        asyncio.run(client.system_one(state={"text": "hi"}, questions={"q": q}))


def test_instructions_that_would_be_trimmed_are_refused(agent):
    client = LayaClient(agent=FakeAgent(agent))
    q = {"type": "noul", "instructions": "Does the text express something? " * 60}
    with pytest.raises(LayaBudgetError, match="instructions are"):
        asyncio.run(client.system_one(state={"text": "hi"}, questions={"q": q}))


def test_a_choice_with_many_options_is_refused_with_the_reason(agent):
    client = LayaClient(agent=FakeAgent(agent))
    q = {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(25)}}
    with pytest.raises(LayaError, match="weak past 20"):
        asyncio.run(client.system_one(state={"text": "hi"}, questions={"q": q}))


def test_a_request_records_what_a_local_engine_spends(agent):
    client = LayaClient(agent=FakeAgent(agent))
    asyncio.run(client.system_one(
        state={"text": "hi"}, questions={"a": {"type": "noul", "instructions": "?"},
                                          "b": {"type": "noul", "instructions": "??"}}))
    assert client.stats.calls == 1 and client.stats.rows == 2
    assert client.stats.input_tokens == 40 and len(client.stats.latencies_ms) == 1


def test_the_steering_host_can_ask_whether_questions_fit_before_spending_a_round(agent):
    client = LayaClient(agent=FakeAgent(agent))
    fine = {"q": {"type": "noul", "instructions": "Does the text express praise?"}}
    wordy = {"q": {"type": "choice", "instructions": "Is the opener hedged? " * 60,
                   "criteria": {"positive": None, "negative": None, "none": None}}}

    client.check_questions(fine, "a short item")                       # no error
    with pytest.raises(LayaBudgetError, match="instructions are"):
        client.check_questions(wordy, "a short item")
    assert client._agent.calls == []                                    # nothing was answered
