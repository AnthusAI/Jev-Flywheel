"""Feature: the deterministic operations behind the steering procedure.

Uses a slice of the real corpus with a simulated human, and a fake Jev client that counts
what it is asked. The reference scorecard's element wordings are all cached, so a proposal
that reuses them costs nothing; a novel element costs one request per labeled item.
"""
import json
import random
import shutil

import pytest

from jev_flywheel.host import FlywheelHost, run_sync
from jev_flywheel.loop import AGREE, DISAGREE, next_question, record_label, refit
from jev_flywheel.items import normalize_label
from jev_flywheel.workspace import Workspace
from tests.loop_test import SCORE, miniature


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("template")
    return Workspace.init(root / "var", miniature(root / "fixtures", n_pool=700, n_test=200))


@pytest.fixture
def workspace(template, tmp_path):
    shutil.copytree(template.root, tmp_path / "var")
    return Workspace(tmp_path / "var")


def label(workspace, count, seed=0, comment="looks sarcastic"):
    rng = random.Random(seed)
    for _ in range(count):
        question = next_question(workspace, SCORE, rng)
        truth = question.item.reference_label
        if normalize_label(truth) == normalize_label(question.result.value):
            record_label(workspace, question, AGREE)
        else:
            record_label(workspace, question, DISAGREE, correct_label=truth, comment=comment)


@pytest.fixture(scope="module")
def labeled_template(template, tmp_path_factory):
    """160 labels and a promoted v2, built once and copied for each test that wants it."""
    root = tmp_path_factory.mktemp("labeled")
    shutil.copytree(template.root, root / "var")
    workspace = Workspace(root / "var")
    label(workspace, 100)
    assert refit(workspace, SCORE).promoted          # v2: a fitted head to attribute from
    label(workspace, 60, seed=1)
    return workspace


@pytest.fixture
def labeled(labeled_template, tmp_path):
    shutil.copytree(labeled_template.root, tmp_path / "var")
    return Workspace(tmp_path / "var")


# Wordings identical to the reference scorecard, so their answers are already cached.
PRAISE = {"key": "praise", "question_type": "noul",
          "instructions": "Does the text express praise or approval of something?"}
CRITICISM = {"key": "criticism", "question_type": "noul",
             "instructions": "Does the text express criticism or disapproval of something?"}
SARCASM = {"key": "sarcasm", "question_type": "noul",
           "instructions": "Is the writer being sarcastic, so the words say the opposite?"}


def reply(**kwargs):
    return json.dumps({"root_cause": "because", **kwargs})


class Usage:
    def model_dump(self):
        return {"input_tokens": 500, "output_tokens": 5}


class Response:
    model = "jev-test"
    usage = Usage()

    def __init__(self, answers):
        self.answers = answers


class Client:
    """Answers correlate with the corpus label, so a new element carries real signal."""

    def __init__(self, workspace):
        self.calls = []
        self._truth = {i.text: i.reference_label for i in workspace.items}

    async def system_one(self, *, state, questions):
        self.calls.append(sorted(questions))
        positive = self._truth.get(state["text"]) == "positive"
        return Response({name: {"type": "noul", "noul": 0.15 if positive else 0.85}
                         for name in questions})


def host_for(workspace, **kwargs):
    return FlywheelHost(workspace, SCORE, **kwargs)


# ---- the briefing ----------------------------------------------------------------

def test_the_briefing_gives_the_agent_the_scorecard_the_counts_and_the_budget(labeled):
    brief = host_for(labeled).briefing()

    assert "name: Sentiment" in brief["scorecard_yaml"]
    summary = brief["summary"]
    assert summary["n_labeled"] == 160
    assert summary["capability_tier"] in {"shrunk", "standard"}
    assert summary["feature_budget"] >= summary["features_in_use"]
    assert summary["classes"] == ["positive", "negative"]
    assert set(summary["label_distribution"]) == {"positive", "negative"}


def test_the_briefing_lists_disagreements_with_the_humans_comments_first(labeled):
    brief = host_for(labeled).briefing()

    assert brief["mismatches"]
    entry = brief["mismatches"][0]
    assert entry["human_comment"] == "looks sarcastic"
    assert entry["we_said"] != entry["human_said"]
    for field in ("item_id", "text", "confidence_when_shown", "element_answers", "top_drivers_now"):
        assert field in entry


def test_commented_disagreements_come_before_uncommented_ones(workspace):
    label(workspace, 60, comment=None)
    label(workspace, 30, seed=2, comment="explained")

    brief = host_for(workspace, max_mismatches=200).briefing()
    commented = [bool(m["human_comment"]) for m in brief["mismatches"]]

    assert commented == sorted(commented, reverse=True)
    assert any(commented) and not all(commented)


def test_the_number_of_disagreements_shown_is_capped(labeled):
    assert len(host_for(labeled, max_mismatches=3).briefing()["mismatches"]) == 3


def test_the_briefing_includes_which_elements_matter_by_permutation_importance(labeled):
    inventory = host_for(labeled).briefing()["element_inventory"]

    assert inventory
    assert {"element", "permutation_importance", "standardized_weight"} <= set(inventory[0])


def test_the_agent_is_never_shown_the_held_out_scoreboard_or_any_test_item(labeled):
    # An agent that saw it would be tuning to the test set one rethink at a time.
    brief = host_for(labeled).briefing()
    serialized = json.dumps(brief, default=str)

    test_texts = [i.text for i in labeled.split("test")]
    assert not any(text in serialized for text in test_texts)
    assert "test" not in json.dumps(brief["summary"]).lower()
    assert "accuracy_on_test" not in serialized


# ---- checking a proposal ---------------------------------------------------------

def test_a_valid_proposal_is_applied_priced_and_diffed_without_spending_anything(labeled):
    host = host_for(labeled)

    check = host.check(reply(add_elements=[SARCASM]))

    assert check["ok"] and not check["noop"]
    assert check["diff"]["added"] == ["sarcasm"]
    assert check["plan"]["requests"] == 160         # every labeled item lacks the new answer
    assert check["plan"]["estimated_input_tokens"] == 160 * 500
    assert check["n_features"] == 2


def test_reusing_already_cached_wordings_costs_nothing(labeled):
    check = host_for(labeled).check(reply(add_elements=[PRAISE, CRITICISM]))

    assert check["ok"]
    assert check["plan"]["requests"] == 0


def test_the_price_of_ten_elements_is_the_price_of_one(labeled):
    # Requests are per item, not per question: the reason a proposal is one batch.
    one = host_for(labeled).check(reply(add_elements=[SARCASM]))
    four = host_for(labeled).check(reply(add_elements=[
        SARCASM, {**SARCASM, "key": "irony", "instructions": "Ironic?"},
        {**SARCASM, "key": "mocking", "instructions": "Mocking tone?"},
        {**SARCASM, "key": "complaint", "instructions": "Complaining?"}]))

    assert one["plan"]["requests"] == four["plan"]["requests"] == 160


def test_a_malformed_proposal_is_refused_with_a_message_for_the_agent(labeled):
    check = host_for(labeled).check("I would add a sarcasm element, I think.")

    assert not check["ok"]
    assert "JSON object" in check["problems"][0]
    assert check["plan"] is None


def test_a_proposal_that_breaks_the_scorecard_rules_is_refused(labeled):
    check = host_for(labeled).check(reply(add_elements=[{**SARCASM, "key": "bad.key"}]))

    assert not check["ok"]
    assert "not valid" in check["problems"][0]


def test_too_many_features_for_the_evidence_is_refused_and_says_what_to_do(workspace):
    label(workspace, 40)                        # ~35 effective labels: room for only a few
    # Each element asks for three features, so four of them is 13 features in all.
    wide = [{**SARCASM, "key": f"e{i}", "instructions": f"Question {i}?",
             "features": [f"e{i}.logit_p", f"e{i}.p", f"e{i}.is_yes"]} for i in range(4)]

    check = host_for(workspace).check(reply(add_elements=wide))

    assert not check["ok"]
    assert "budget" in check["problems"][0] and "Retire" in check["problems"][0]


def test_an_empty_proposal_is_reported_as_a_no_op(labeled):
    check = host_for(labeled).check(reply())

    assert check["ok"] and check["noop"]


def test_a_lua_style_table_is_accepted_as_a_proposal(labeled):
    class Table(dict):
        pass

    check = host_for(labeled).check(Table(root_cause="x", add_elements=[Table(SARCASM)]))

    assert check["ok"] and check["diff"]["added"] == ["sarcasm"]


# ---- evaluating ------------------------------------------------------------------

def test_evaluating_without_a_checked_proposal_is_an_error(labeled):
    assert host_for(labeled).evaluate()["status"] == "error"


def test_a_candidate_that_needs_new_answers_is_not_evaluated_unless_spending_is_allowed(labeled):
    client = Client(labeled)
    host = host_for(labeled, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))

    result = host.evaluate()

    assert result["status"] == "needs_spend"
    assert result["requests"] == 160
    assert client.calls == []


def test_evaluating_spends_exactly_one_request_per_labeled_item_for_only_the_new_question(labeled):
    client = Client(labeled)
    host = host_for(labeled, allow_spend=True, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))

    result = host.evaluate()

    assert result["status"] == "fitted"
    assert len(client.calls) == 160
    assert all(call == ["sentiment.sarcasm"] for call in client.calls)


def test_a_candidate_whose_answers_are_all_cached_is_evaluated_for_free(labeled):
    client = Client(labeled)
    host = host_for(labeled, client_factory=lambda: client)     # spending NOT allowed
    host.check(reply(add_elements=[PRAISE, CRITICISM]))

    result = host.evaluate()

    assert result["status"] == "fitted"
    assert client.calls == []
    assert set(result["candidate"]) == {"accuracy", "ece", "brier", "n"}


def test_a_genuinely_informative_element_earns_promotion(labeled):
    client = Client(labeled)
    host = host_for(labeled, allow_spend=True, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))

    result = host.evaluate()

    assert result["promote"], result["reasons"]
    assert result["candidate"]["accuracy"] > result["incumbent"]["accuracy"]


def test_a_candidate_with_too_few_labels_is_held_not_fit(workspace):
    label(workspace, 10)
    host = host_for(workspace)
    host.check(reply(add_elements=[PRAISE]))

    assert host.evaluate()["status"] in {"held", "refused"}


# ---- applying --------------------------------------------------------------------

def test_an_evaluated_and_better_candidate_becomes_a_new_scorecard_version(labeled):
    client = Client(labeled)
    host = host_for(labeled, allow_spend=True, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))
    assert host.evaluate()["promote"]
    before = labeled.version

    result = host.apply({"model": "kimi", "note": "from the agent"})

    assert result["version"] == before + 1
    entry = labeled.lineage()[-1]
    assert entry["kind"] == "steer"
    assert entry["provenance"]["proposal"]["added"] == ["sarcasm"]
    assert entry["provenance"]["root_cause"] == "because"
    assert entry["provenance"]["model"] == "kimi"
    assert "sentiment.sarcasm" in labeled.scorecard().questions()
    assert "sarcasm.logit_p" in labeled.scorecard().score(SCORE).decision.features


def test_the_applied_scorecard_carries_fitted_weights_and_a_calibration(labeled):
    client = Client(labeled)
    host = host_for(labeled, allow_spend=True, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))
    host.evaluate()
    host.apply()

    decision = labeled.scorecard().score(SCORE).decision

    assert decision.weights and "sarcasm.logit_p" in next(iter(decision.weights.values()))
    assert decision.calibration and decision.provenance["fit_id"].startswith("dh-")


def test_nothing_can_be_applied_that_was_not_evaluated(labeled):
    host = host_for(labeled)
    host.check(reply(add_elements=[SARCASM]))          # checked, never evaluated

    with pytest.raises(RuntimeError, match="evaluated"):
        host.apply()
    assert labeled.version == 2


def test_a_candidate_that_did_not_beat_the_incumbent_cannot_be_applied(labeled):
    host = host_for(labeled)
    host.check(reply())                                # a no-op: cannot beat itself
    result = host.evaluate()
    assert not result.get("promote")

    with pytest.raises(RuntimeError, match="evaluated"):
        host.apply()


def test_applying_twice_commits_once(labeled):
    client = Client(labeled)
    host = host_for(labeled, allow_spend=True, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))
    host.evaluate()

    first = host.apply()
    second = host.apply()

    assert second == {"version": first["version"], "already_applied": True}
    assert labeled.version == first["version"]


def test_a_new_check_discards_a_previous_candidate(labeled):
    client = Client(labeled)
    host = host_for(labeled, allow_spend=True, client_factory=lambda: client)
    host.check(reply(add_elements=[SARCASM]))
    host.evaluate()

    host.check("not even json")

    with pytest.raises(RuntimeError):
        host.apply()


def test_run_sync_works_from_inside_a_running_event_loop():
    # Tactus calls host methods from inside its own loop, where asyncio.run would raise.
    import asyncio

    async def inner():
        return 7

    async def outer():
        return run_sync(inner())

    assert asyncio.run(outer()) == 7


# ---- a failed top-up must say so ---------------------------------------------------

class Broken:
    """Jev is unreachable: every request fails."""

    async def system_one(self, *, state, questions):
        raise RuntimeError("no API key configured")


def test_a_failed_top_up_says_it_failed_instead_of_blaming_the_label_count(labeled):
    # Regression: with the key missing, every request failed, the fit then saw zero
    # labeled items, and the message read "0 effective labels, 30 more are needed".
    host = host_for(labeled, allow_spend=True, client_factory=lambda: Broken())
    host.check(reply(add_elements=[SARCASM]))

    result = host.evaluate()

    assert result["status"] == "top_up_failed"
    assert not result["promote"]
    assert result["failed"] == 160
    assert "failed for 160 of 160" in result["reason"]
    assert "no API key configured" in result["reason"]
    assert "Nothing was fit" in result["reason"]


def test_a_few_failed_requests_do_not_abort_the_evaluation(labeled):
    class Flaky(Client):
        async def system_one(self, *, state, questions):
            if len(self.calls) in (3, 9):
                self.calls.append(["boom"])
                raise RuntimeError("transient")
            return await super().system_one(state=state, questions=questions)

    host = host_for(labeled, allow_spend=True, client_factory=lambda: Flaky(labeled))
    host.check(reply(add_elements=[SARCASM]))

    assert host.evaluate()["status"] == "fitted"


# ---- serving cost is priced too, not just evaluation cost --------------------------

def test_the_check_prices_serving_the_candidate_on_every_item_not_just_evaluating_it(labeled):
    check = host_for(labeled).check(reply(add_elements=[SARCASM]))

    plan = check["plan"]
    assert plan["requests"] == 160                      # evaluation: labeled items only
    assert plan["serving_requests"] == len(labeled.items)   # serving: every item
    assert f"{len(labeled.items):,} requests in all" in check["summary_text"]


def test_rewording_the_holistic_question_is_flagged_as_making_stored_answers_stale(labeled):
    check = host_for(labeled).check(reply(
        reword_elements=[{"key": "holistic", "instructions": "What is the real sentiment?"}]))

    assert "stale" in check["summary_text"]
    assert check["plan"]["serving_requests"] == len(labeled.items)
