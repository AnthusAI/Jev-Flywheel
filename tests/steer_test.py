"""Feature: the steering procedure, run for real through Tactus with a scripted analyst.

The language model is the one thing replaced, by scripted replies. The Tactus runtime, the
Lua procedure, the Python host, the fit, the approvals and the commit all run for real, so
these specs exercise the actual loop and need no API keys.
"""
import json
import shutil

import pytest

from jev_flywheel.steer import ScriptedApprover, SteerError, render_source, run_steering
from jev_flywheel.workspace import Workspace
from tests.host_test import (  # noqa: F401  (fixtures)
    SARCASM, Client, Response, labeled, labeled_template, reply, template, workspace)
from tests.loop_test import SCORE

pytest.importorskip("tactus")


class Silent(Client):
    """Answers that carry no signal, so a proposed element cannot help."""

    async def system_one(self, *, state, questions):
        from tests.host_test import Response
        self.calls.append(sorted(questions))
        return Response({name: {"type": "noul", "noul": 0.5} for name in questions})


def steer(workspace, replies, *, approvals=(True,), client=None, allow_spend=True, **kwargs):
    client = client or Client(workspace)
    approver = ScriptedApprover(approvals)
    outcome = run_steering(
        workspace, SCORE, allow_spend=allow_spend, client_factory=lambda: client,
        hitl_handler=approver, mock_replies=replies, **kwargs)
    return outcome, client, approver


def test_the_procedure_source_has_its_placeholders_filled_in():
    source = render_source(provider="bedrock", model="us.moonshotai.kimi-k3", max_tokens=9000)

    assert 'provider = "bedrock"' in source
    assert 'model = "us.moonshotai.kimi-k3"' in source
    assert "max_tokens = 9000" in source
    assert "{{" not in source


def test_a_good_proposal_is_evaluated_approved_and_applied_as_a_new_version(labeled):
    before = labeled.version

    outcome, client, approver = steer(labeled, [reply(add_elements=[SARCASM])])

    assert outcome.decision == "promoted"
    assert outcome.detail["version"] == before + 1
    assert labeled.version == before + 1
    assert labeled.lineage()[-1]["kind"] == "steer"
    assert "sentiment.sarcasm" in labeled.scorecard().questions()
    assert outcome.detail["root_cause"] == "because"


def test_the_human_is_shown_the_change_its_cost_and_the_projected_effect(labeled):
    _, _, approver = steer(labeled, [reply(add_elements=[SARCASM])])

    assert len(approver.asked) == 1
    shown = approver.asked[0]
    assert "Why: because" in shown
    assert "Changes: add sarcasm" in shown
    assert "sarcasm.logit_p" in shown
    assert "candidate:" in shown and "incumbent:" in shown
    assert "accuracy" in shown and "ECE" in shown


def test_a_human_who_says_no_leaves_the_scorecard_untouched(labeled):
    before = labeled.version

    outcome, _, approver = steer(labeled, [reply(add_elements=[SARCASM])], approvals=(False,))

    assert outcome.decision == "rejected_by_human"
    assert labeled.version == before
    assert len(approver.asked) == 1


def test_a_proposal_of_no_change_costs_nothing_and_asks_nobody(labeled):
    outcome, client, approver = steer(labeled, [reply()])

    assert outcome.decision == "no_change_proposed"
    assert client.calls == [] and approver.asked == []


def test_a_candidate_that_cannot_beat_the_incumbent_never_reaches_the_human(labeled):
    # The new element's answers carry no signal, so the candidate is no better.
    outcome, client, approver = steer(
        labeled, [reply(add_elements=[SARCASM])], client=Silent(labeled))

    assert outcome.decision == "rejected_by_metrics"
    assert approver.asked == []
    assert len(client.calls) == 160            # it did spend, once, to find that out
    assert "Not promoted" in outcome.detail["reasons"]


def test_a_malformed_reply_is_repaired_once_at_no_cost(labeled):
    outcome, client, _ = steer(labeled, ["I think you should add sarcasm.",
                                         reply(add_elements=[SARCASM])])

    assert outcome.decision == "promoted"
    assert len(client.calls) == 160           # spent once, on the repaired proposal only


def test_a_reply_that_stays_malformed_is_abandoned_without_spending(labeled):
    before = labeled.version

    outcome, client, approver = steer(labeled, ["not json", "still not json"])

    assert outcome.decision == "invalid_proposal"
    assert "JSON object" in outcome.detail["problem"]
    assert client.calls == [] and approver.asked == [] and labeled.version == before


def test_repairs_can_be_switched_off(labeled):
    outcome, _, _ = steer(labeled, ["not json", reply(add_elements=[SARCASM])], max_revisions=0)

    assert outcome.decision == "invalid_proposal"


def test_a_costly_proposal_asks_before_spending_and_a_no_spends_nothing(labeled):
    outcome, client, approver = steer(
        labeled, [reply(add_elements=[SARCASM])], approvals=(False,), max_auto_requests=10)

    assert outcome.decision == "declined_spend"
    assert "needs 160 Jev requests" in approver.asked[0]
    assert client.calls == []


def test_a_costly_proposal_that_is_approved_goes_on_to_the_normal_review(labeled):
    outcome, client, approver = steer(
        labeled, [reply(add_elements=[SARCASM])], approvals=(True, True), max_auto_requests=10)

    assert outcome.decision == "promoted"
    assert len(approver.asked) == 2            # once for the spend, once for the change
    assert len(client.calls) == 160


def test_a_run_that_may_not_call_jev_stops_at_the_price_instead_of_spending(labeled):
    outcome, client, _ = steer(
        labeled, [reply(add_elements=[SARCASM])], allow_spend=False)

    assert outcome.decision == "needs_spend"
    assert outcome.detail["requests"] == 160
    assert client.calls == []


def test_a_proposal_reusing_cached_wordings_is_evaluated_with_no_jev_access_at_all(labeled):
    from tests.host_test import CRITICISM, PRAISE

    outcome, client, _ = steer(
        labeled, [reply(add_elements=[PRAISE, CRITICISM])], allow_spend=False)

    assert outcome.decision in {"promoted", "rejected_by_metrics"}
    assert client.calls == []


def test_the_agent_cannot_smuggle_weights_in_the_fit_sets_them(labeled):
    smuggled = reply(add_elements=[SARCASM], weights={"positive": {"intercept": 999.0}},
                     calibration={"method": "none"})

    outcome, _, _ = steer(labeled, [smuggled])

    assert outcome.decision == "promoted"
    weights = labeled.scorecard().score(SCORE).decision.weights
    assert weights["positive"]["intercept"] != 999.0
    assert abs(weights["positive"]["intercept"]) < 10


def test_every_round_is_recorded_as_a_rethink_event_whatever_it_decided(labeled):
    steer(labeled, [reply()])
    steer(labeled, [reply(add_elements=[SARCASM])], approvals=(False,))

    events = labeled.events("rethink")

    assert [e["decision"] for e in events] == ["no_change_proposed", "rejected_by_human"]
    assert all(e["score_name"] == SCORE for e in events)


def test_a_rethink_resets_the_steering_cooldown(labeled):
    steer(labeled, [reply()])

    state = labeled.steering_state(SCORE)

    assert state.labels_since_rethink == 0
    assert state.commented_mismatches_since_rethink == 0


def test_the_analyst_is_shown_the_briefing_the_host_built(labeled):
    # The prompt lives in the procedure; check what it puts in front of the model by
    # asking a spy to be the analyst.
    from tests.host_test import FlywheelHost  # noqa: F401
    import jev_flywheel.steer as steer_module

    seen = {}
    original = steer_module.FlywheelHost.briefing

    def spy(self):
        seen["brief"] = original(self)
        return seen["brief"]

    steer_module.FlywheelHost.briefing = spy
    try:
        steer(labeled, [reply()])
    finally:
        steer_module.FlywheelHost.briefing = original

    assert seen["brief"]["mismatches"]
    assert "held" not in json.dumps(seen["brief"]["summary"]).lower()


def test_a_failure_inside_the_procedure_raises_an_error_that_says_why(labeled):
    with pytest.raises(SteerError):
        run_steering(labeled, "No Such Score", hitl_handler=ScriptedApprover(),
                     mock_replies=[reply()])


def test_a_scripted_round_never_reaches_a_real_model(labeled, monkeypatch):
    # Agents are built while the procedure is parsed, so forgetting to enable mocking
    # before that quietly builds one against the real model: a live, paid call. This is
    # the regression guard for that mistake.
    import litellm

    def boom(*args, **kwargs):
        raise AssertionError("a scripted round called a real model")

    async def aboom(*args, **kwargs):
        raise AssertionError("a scripted round called a real model")

    monkeypatch.setattr(litellm, "completion", boom)
    monkeypatch.setattr(litellm, "acompletion", aboom)

    outcome, _, _ = steer(labeled, [reply()])

    assert outcome.decision == "no_change_proposed"


def test_steering_without_tactus_explains_how_to_install_it(labeled, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("tactus"):
            raise ImportError("no tactus")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(SteerError, match=r"jev-flywheel\[steer\]"):
        run_steering(labeled, SCORE, hitl_handler=object(), mock_replies=[reply()])


def test_a_bedrock_region_can_be_set_per_model_and_defaults_to_us_east_1(monkeypatch):
    from jev_flywheel.steer import apply_region

    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    apply_region("bedrock", None)
    assert __import__("os").environ["AWS_DEFAULT_REGION"] == "us-east-1"

    apply_region("bedrock", "us-west-2")          # e.g. Qwen3 Coder 480B is not in us-east-1
    assert __import__("os").environ["AWS_DEFAULT_REGION"] == "us-west-2"

    apply_region("bedrock", None)                 # an already-chosen region is kept
    assert __import__("os").environ["AWS_DEFAULT_REGION"] == "us-west-2"


def test_the_region_is_ignored_for_other_providers(monkeypatch):
    from jev_flywheel.steer import apply_region

    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    apply_region("openai", "us-west-2")

    assert "AWS_DEFAULT_REGION" not in __import__("os").environ


def test_an_evaluation_that_could_not_run_records_why_in_the_event(labeled):
    class Broken2(Client):
        async def system_one(self, *, state, questions):
            raise RuntimeError("engine fell over")

    outcome, _, _ = steer(labeled, [reply(add_elements=[SARCASM])], client=Broken2(labeled))

    event = labeled.events("rethink")[-1]
    assert outcome.decision == "not_evaluable"
    assert event["decision"] == "not_evaluable"
    assert event["status"] == "top_up_failed"
    assert "engine fell over" in event["reason"]


# ---- the gender-invariance gate (studies/PREREGISTERED.md's J2/L2 arms) --------------------
#
# The sentiment fixture's own texts (AI-generated workplace/sports snippets) have no personal
# subject and so no pronoun swap_gender can touch -- true of the corpus these specs otherwise
# share, and not something to fix here. These tests append an unambiguous sentence with a
# pronoun to each labeled item's text before steering, so the swap has something to flip.

GENDERED_TONE = {"key": "gendered_tone", "question_type": "noul",
                 "instructions": "Does the text read as forceful or assertive?"}


def _give_labeled_items_a_pronoun(workspace):
    """Append "He said this himself." to every labeled item whose reference label is
    "positive" and "She said this herself." to every "negative" one (in place), so a pronoun
    a client can key off of is coupled to the label it is trying to predict -- otherwise an
    element that reads the pronoun would have no signal to lose and the ordinary fit test
    would reject it whether or not the invariance gate did. Returns the labeled item ids."""
    ids = sorted({f.item_id for f in workspace.feedback()})
    for item_id in ids:
        item = workspace.item(item_id)
        pronoun_sentence = (" He said this himself." if item.reference_label == "positive"
                            else " She said this herself.")
        item.text = item.text + pronoun_sentence
    return ids


_PRONOUN_SUFFIX = __import__("re").compile(r" (He|She) said this (himself|herself)\.$")


class GenderReader(Client):
    """Every element answers from the corpus label except ``gendered_tone``, which reads
    a masculine pronoun straight off the text -- so its own answer flips whenever
    ``swap_gender`` changes one, and every other element's does not.

    ``_truth`` is keyed by the *unswapped* text (``Client.__init__``'s dict), so looking the
    corpus label up under the swapped twin's text -- which differs only in the pronoun
    ``_give_labeled_items_a_pronoun`` appended -- would always miss. Stripping that suffix
    before the lookup is what makes "positive" a property of the corpus label alone, the way a
    real engine reading actual content would answer it; the miss otherwise has nothing to do
    with the invariance gate under test.
    """

    def __init__(self, workspace):
        super().__init__(workspace)
        self._truth = {_PRONOUN_SUFFIX.sub("", text): label for text, label in self._truth.items()}

    async def system_one(self, *, state, questions):
        import re

        self.calls.append(sorted(questions))
        stripped = _PRONOUN_SUFFIX.sub("", state["text"])
        positive = self._truth.get(stripped) == "positive"
        masculine = bool(re.search(r"\b(he|him|his|himself)\b", state["text"], re.IGNORECASE))
        return Response({
            name: {"type": "noul",
                  "noul": (0.9 if masculine else 0.1) if name.endswith("gendered_tone")
                  else (0.15 if positive else 0.85)}
            for name in questions})


def test_the_gate_rejects_an_element_that_reads_gender(labeled):
    _give_labeled_items_a_pronoun(labeled)

    outcome, client, approver = steer(
        labeled, [reply(add_elements=[GENDERED_TONE])], client=GenderReader(labeled),
        invariance_max_flip_rate=0.02)

    assert outcome.decision == "rejected_by_metrics"
    assert "invariance gate" in outcome.detail["reasons"]      # evaluation.summary_text, a string
    assert approver.asked == []                 # never reached a human: gated first
    # The gate's own number is on the outcome too, not just summarized as prose -- this is
    # what a study recording every proposal's flip rate on the labeled items reads.
    assert outcome.invariance_flip_rates["sentiment.gendered_tone"] > 0.02


def test_the_gate_lets_through_an_element_that_does_not_read_gender(labeled):
    _give_labeled_items_a_pronoun(labeled)

    # SARCASM's answers correlate with the corpus label only, never with pronouns, so its
    # own flip rate under the swap is zero regardless of the pronoun added above.
    outcome, client, _ = steer(
        labeled, [reply(add_elements=[SARCASM])], client=GenderReader(labeled),
        invariance_max_flip_rate=0.02)

    assert outcome.decision == "promoted"
    assert outcome.invariance_flip_rates["sentiment.sarcasm"] == 0.0


def test_with_no_gate_the_same_gendered_element_is_promoted(labeled):
    _give_labeled_items_a_pronoun(labeled)

    # invariance_max_flip_rate=None (the default) must reproduce the pre-gate behaviour
    # exactly: the same element that the gate rejects above is free to be promoted here.
    outcome, _, _ = steer(
        labeled, [reply(add_elements=[GENDERED_TONE])], client=GenderReader(labeled))

    assert outcome.decision == "promoted"


def test_the_analyst_is_told_the_gate_exists_when_it_is_on():
    from jev_flywheel.steer import render_source

    off = render_source()
    on = render_source(invariance_max_flip_rate=0.02)

    assert "gate" not in off.lower()
    assert "gate will reject" in on
    assert "2%" in on
    assert "{{" not in on
