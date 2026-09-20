"""Feature: the answer cache is keyed per question, so a new element is cheap.

Adding an element must cost a top-up of just that question. If it voided every
cached answer, the flywheel's whole premise -- propose better elements -- would
cost a full Jev pass per proposal.
"""
import json

import pytest

from jev_flywheel.answers import AnswerCache, import_answers_jsonl, question_hash
from jev_flywheel.items import Item
from jev_flywheel.jev import JevSession

PRAISE = {"type": "noul", "instructions": "Praise?"}
IRONY = {"type": "noul", "instructions": "Irony?"}
SENTIMENT = {"type": "choice", "instructions": "Sentiment?",
             "criteria": {"positive": None, "negative": None}}
QUESTIONS = {"Sentiment": SENTIMENT, "s.praise": PRAISE}


class Usage:
    def model_dump(self):
        return {"input_tokens": 100, "output_tokens": 5}


class Response:
    model = "jev-test"
    usage = Usage()

    def __init__(self, answers):
        self.answers = answers


class Client:
    """Answers whatever it is asked, and records exactly what was asked."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    async def system_one(self, *, state, questions):
        self.calls.append((state["text"], sorted(questions)))
        if state["text"] in self.fail_on:
            raise RuntimeError("boom")
        return Response({name: {"type": "noul", "noul": 0.9} if q["type"] == "noul"
                         else {"type": "choice", "choice": "positive",
                               "probabilities": {"positive": 0.9, "negative": 0.1}}
                         for name, q in questions.items()})


def session(client):
    return JevSession(client_factory=lambda: client)


def items(*ids):
    return [Item(id=i, text=f"text {i}") for i in ids]


def test_an_answer_is_found_by_item_question_and_wording():
    cache = AnswerCache()
    cache.put("a", "s.praise", PRAISE, {"type": "noul", "noul": 0.9})

    assert cache.get("a", "s.praise", PRAISE)["noul"] == 0.9
    assert cache.get("b", "s.praise", PRAISE) is None


def test_reworded_question_is_a_miss_so_it_gets_re_asked():
    # Keying on the body hash is what makes a reworded element re-ask while
    # every other answer is reused.
    cache = AnswerCache()
    cache.put("a", "s.praise", PRAISE, {"type": "noul", "noul": 0.9})

    reworded = {"type": "noul", "instructions": "Does the text express approval?"}

    assert question_hash(reworded) != question_hash(PRAISE)
    assert cache.get("a", "s.praise", reworded) is None


def test_missing_reports_only_the_questions_with_no_answer_on_file():
    cache = AnswerCache()
    cache.put("a", "s.praise", PRAISE, {"type": "noul", "noul": 0.9})

    gap = cache.missing("a", {**QUESTIONS, "s.irony": IRONY})

    assert set(gap) == {"Sentiment", "s.irony"}


def test_answers_for_is_none_unless_every_question_is_answered():
    # Returning a partial mapping by accident would let a fit train on a
    # half-empty row. Partial access has to be asked for by name.
    cache = AnswerCache()
    cache.put("a", "Sentiment", SENTIMENT, {"type": "choice", "choice": "positive",
                                            "probabilities": {"positive": 1.0, "negative": 0.0}})

    assert cache.answers_for("a", QUESTIONS) is None
    assert set(cache.partial_answers_for("a", QUESTIONS)) == {"Sentiment"}


def test_the_cache_persists_and_reloads(tmp_path):
    path = tmp_path / "answers.jsonl"
    AnswerCache(path).put("a", "s.praise", PRAISE, {"type": "noul", "noul": 0.7}, "jev-x")

    reloaded = AnswerCache(path)

    assert reloaded.get("a", "s.praise", PRAISE)["noul"] == 0.7
    assert reloaded.model == "jev-x"
    assert len(reloaded) == 1


def test_the_cache_stores_item_ids_and_never_the_text(tmp_path):
    # Not a second copy of the corpus, so it can be committed as a fixture.
    path = tmp_path / "answers.jsonl"
    AnswerCache(path).put("a", "s.praise", PRAISE, {"type": "noul", "noul": 0.7})

    assert "text" not in json.loads(path.read_text().splitlines()[0])


def test_score_answers_with_integer_keys_are_stored_as_strings():
    cache = AnswerCache()
    question = {"type": "score", "criteria": ["a", "b"]}
    cache.put("a", "s.level", question, {"type": "score", "score": 1.0,
                                         "probabilities": {0: 0.2, 1: 0.8}})

    assert cache.get("a", "s.level", question)["probabilities"] == {"0": 0.2, "1": 0.8}


def test_a_plan_prices_a_top_up_without_spending_anything():
    cache = AnswerCache()
    cache.put("a", "Sentiment", SENTIMENT, {"type": "choice", "choice": "positive",
                                            "probabilities": {"positive": 1.0, "negative": 0.0}})
    cache.put("a", "s.praise", PRAISE, {"type": "noul", "noul": 0.9})

    plan = cache.plan(["a", "b"], QUESTIONS)

    assert plan.requests == 1                # only "b" needs anything
    assert plan.missing_answers == 2
    assert plan.items_needing == ["b"]
    assert not plan.is_free
    assert cache.plan(["a"], QUESTIONS).is_free


def test_ten_new_elements_cost_the_same_requests_as_one():
    # The reason a steering cycle must batch every proposal into one change:
    # requests are per item, not per question.
    cache = AnswerCache()
    for item_id in ("a", "b", "c"):
        for name, q in QUESTIONS.items():
            cache.put(item_id, name, q, {"type": "noul", "noul": 0.5})
    one = {**QUESTIONS, "s.e0": {"type": "noul", "instructions": "0?"}}
    ten = {**QUESTIONS, **{f"s.e{i}": {"type": "noul", "instructions": f"{i}?"}
                           for i in range(10)}}

    assert cache.plan(["a", "b", "c"], one).requests == 3
    assert cache.plan(["a", "b", "c"], ten).requests == 3
    assert cache.plan(["a", "b", "c"], ten).missing_answers == 30


async def test_fill_asks_one_request_per_item_carrying_only_what_is_missing():
    client = Client()
    cache = AnswerCache()
    cache.put("a", "Sentiment", SENTIMENT, {"type": "choice", "choice": "positive",
                                            "probabilities": {"positive": 1.0, "negative": 0.0}})

    report = await cache.fill(session(client), items("a", "b"), QUESTIONS)

    asked = dict(client.calls)
    assert asked["text a"] == ["s.praise"]              # only the gap
    assert asked["text b"] == ["Sentiment", "s.praise"]
    assert report.requested == 2
    assert cache.answers_for("a", QUESTIONS) is not None
    assert cache.answers_for("b", QUESTIONS) is not None


async def test_a_complete_item_costs_nothing():
    client = Client()
    cache = AnswerCache()
    await cache.fill(session(client), items("a"), QUESTIONS)
    client.calls.clear()

    report = await cache.fill(session(client), items("a"), QUESTIONS)

    assert client.calls == []
    assert report.already_complete == 1


async def test_adding_an_element_tops_up_only_that_question():
    # The central claim: a new element does not re-ask everything on file.
    client = Client()
    cache = AnswerCache()
    await cache.fill(session(client), items("a", "b"), QUESTIONS)
    client.calls.clear()

    await cache.fill(session(client), items("a", "b"), {**QUESTIONS, "s.irony": IRONY})

    assert sorted(name for _, names in client.calls for name in names) == ["s.irony", "s.irony"]
    assert len(client.calls) == 2


async def test_retiring_an_element_costs_nothing_and_its_answers_are_ignored():
    client = Client()
    cache = AnswerCache()
    await cache.fill(session(client), items("a"), QUESTIONS)
    client.calls.clear()

    fewer = {"Sentiment": SENTIMENT}
    report = await cache.fill(session(client), items("a"), fewer)

    assert client.calls == []
    assert report.already_complete == 1
    assert set(cache.answers_for("a", fewer)) == {"Sentiment"}


async def test_a_failed_item_is_counted_not_raised_and_the_rest_complete():
    client = Client(fail_on={"text b"})
    cache = AnswerCache()

    report = await cache.fill(session(client), items("a", "b", "c"), QUESTIONS)

    assert report.failures == 1
    assert report.requested == 2
    assert cache.answers_for("a", QUESTIONS) is not None
    assert cache.answers_for("b", QUESTIONS) is None


async def test_a_rerun_after_a_failure_resumes_only_the_failed_item(tmp_path):
    path = tmp_path / "answers.jsonl"
    await AnswerCache(path).fill(session(Client(fail_on={"text b"})),
                                 items("a", "b"), QUESTIONS)

    client = Client()
    report = await AnswerCache(path).fill(session(client), items("a", "b"), QUESTIONS)

    assert [text for text, _ in client.calls] == ["text b"]
    assert report.failures == 0


async def test_fill_reports_token_usage_and_progress():
    seen = []

    async def progress(done, total):
        seen.append((done, total))

    report = await AnswerCache().fill(session(Client()), items("a", "b"), QUESTIONS,
                                      on_progress=progress)

    assert report.input_tokens == 200
    assert report.output_tokens == 10
    assert seen[-1] == (2, 2)


def test_an_extract_imports_under_the_current_wording(tmp_path):
    source = tmp_path / "extract.jsonl"
    source.write_text(json.dumps({
        "id": "a", "model": "jev-1.13.0",
        "answers": {"Sentiment": {"type": "choice", "choice": "positive",
                                  "probabilities": {"positive": 0.9, "negative": 0.1}},
                    "s.praise": {"type": "noul", "noul": 0.8},
                    "s.retired": {"type": "noul", "noul": 0.1}}}) + "\n")
    cache = AnswerCache()

    stored = import_answers_jsonl(source, QUESTIONS, cache)

    assert stored == 2                      # the retired question is skipped
    assert cache.answers_for("a", QUESTIONS) is not None
    assert cache.model == "jev-1.13.0"


def test_an_imported_answer_is_stale_once_its_question_is_reworded(tmp_path):
    source = tmp_path / "extract.jsonl"
    source.write_text(json.dumps({"id": "a", "answers": {
        "s.praise": {"type": "noul", "noul": 0.8}}}) + "\n")
    cache = AnswerCache()
    import_answers_jsonl(source, {"s.praise": PRAISE}, cache)

    reworded = {"s.praise": {"type": "noul", "instructions": "Something else entirely?"}}

    assert cache.answers_for("a", reworded) is None
