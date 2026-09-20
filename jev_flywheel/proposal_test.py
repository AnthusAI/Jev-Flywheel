"""Feature: the steering agent proposes edits; code applies them.

The agent has nowhere to put a weight, and a proposal is one batch by construction.
Malformed proposals fail with a message that can be fed straight back to the agent.
"""
import json

import pytest

from jev_flywheel.proposal import (
    MAX_NEW_ELEMENTS,
    Proposal,
    ProposalError,
    apply_proposal,
    default_features,
    diff_summary,
    parse_proposal,
)
from jev_flywheel.scorecard import Scorecard

CARD = """
name: Card
version: 3
scores:
  - name: Sentiment
    key: sentiment
    question_type: choice
    instructions: "Overall sentiment?"
    criteria: {positive: null, negative: null}
    elements:
      - {key: praise, question_type: noul, instructions: "Praise?"}
      - {key: criticism, question_type: noul, instructions: "Criticism?"}
    decision:
      model: multinomial_logistic
      classes: ["positive", "negative"]
      features: [self.holistic.clr.positive, praise.logit_p, criticism.logit_p]
      parameters:
        weights:
          positive: {intercept: 0.4, self.holistic.clr.positive: 1.2, praise.logit_p: 0.9,
                     criticism.logit_p: -0.8}
      calibration: {method: temperature, temperature: 1.4,
                    raw_confidence: [0.0, 1.0], calibrated_confidence: [0.0, 0.9]}
      provenance: {fit_id: dh-old, tier: shrunk}
"""

ADD_SARCASM = {"key": "sarcasm", "question_type": "noul", "instructions": "Is it sarcastic?"}


def card():
    return Scorecard.from_yaml(CARD)


def parse(**kwargs):
    return parse_proposal(json.dumps({"root_cause": "why", **kwargs}))


def applied(**kwargs):
    return apply_proposal(card(), "Sentiment", parse(**kwargs))


# ---- parsing ---------------------------------------------------------------------

def test_a_plain_json_reply_is_parsed():
    proposal = parse(add_elements=[ADD_SARCASM], retire_elements=["criticism"])

    assert proposal.root_cause == "why"
    assert [a.key for a in proposal.add] == ["sarcasm"]
    assert proposal.retire == ["criticism"]


def test_json_wrapped_in_prose_and_a_code_fence_is_still_found():
    reply = ('Here is my analysis.\n```json\n'
             + json.dumps({"root_cause": "x", "add_elements": [ADD_SARCASM]})
             + "\n```\nHope that helps!")

    assert parse_proposal(reply).add[0].key == "sarcasm"


def test_json_with_leading_prose_but_no_fence_is_still_found():
    reply = "Sure! " + json.dumps({"root_cause": "x", "retire_elements": ["praise"]}) + " Done."

    assert parse_proposal(reply).retire == ["praise"]


def test_an_already_parsed_object_is_accepted_too():
    assert parse_proposal({"root_cause": "x", "add_elements": [ADD_SARCASM]}).add


def test_a_reply_with_no_json_is_rejected_in_words_the_agent_can_act_on():
    with pytest.raises(ProposalError, match="single JSON object"):
        parse_proposal("I think you should add a sarcasm element.")


def test_an_empty_proposal_is_a_no_op():
    proposal = parse()

    assert proposal.is_noop
    assert proposal.summary() == {"added": [], "retired": [], "reworded": []}


def test_an_added_element_needs_a_key_a_type_and_a_question():
    for missing in ("key", "question_type", "instructions"):
        entry = {k: v for k, v in ADD_SARCASM.items() if k != missing}
        with pytest.raises(ProposalError, match=missing):
            parse(add_elements=[entry])


def test_a_choice_or_score_element_needs_criteria():
    with pytest.raises(ProposalError, match="needs 'criteria'"):
        parse(add_elements=[{**ADD_SARCASM, "question_type": "choice"}])


def test_choice_criteria_may_be_given_as_a_list_and_become_a_mapping():
    proposal = parse(add_elements=[{**ADD_SARCASM, "question_type": "choice",
                                    "criteria": ["yes_sarcasm", "no_sarcasm"]}])

    assert proposal.add[0].criteria == {"yes_sarcasm": None, "no_sarcasm": None}


def test_more_new_elements_than_the_cap_is_rejected_with_the_reason():
    many = [{**ADD_SARCASM, "key": f"e{i}"} for i in range(MAX_NEW_ELEMENTS + 1)]

    with pytest.raises(ProposalError, match="feature budget"):
        parse(add_elements=many)


def test_one_element_may_not_be_both_retired_and_reworded():
    with pytest.raises(ProposalError, match="more than once"):
        parse(retire_elements=["praise"],
              reword_elements=[{"key": "praise", "instructions": "new"}])


def test_a_proposal_has_no_field_for_weights_so_an_agent_cannot_author_them():
    # Extra keys are simply ignored: there is nowhere for them to go.
    proposal = parse(add_elements=[ADD_SARCASM], weights={"positive": {"intercept": 99}},
                     calibration={"method": "none"})

    assert not hasattr(proposal, "weights") and not hasattr(proposal, "calibration")


# ---- default features ------------------------------------------------------------

def test_a_yes_no_element_gets_one_feature():
    assert default_features("sarcasm", "noul", None) == ["sarcasm.logit_p"]


def test_a_choice_element_drops_one_option_because_the_log_ratios_sum_to_zero():
    features = default_features("tone", "choice", {"calm": None, "angry": None, "sad": None})

    assert features == ["tone.clr.calm", "tone.clr.angry"]


def test_a_score_element_gets_its_expected_level():
    assert default_features("intensity", "score", ["a", "b", "c"]) == ["intensity.expected_level"]


def test_an_unknown_question_type_is_rejected():
    with pytest.raises(ProposalError, match="question_type"):
        default_features("x", "freeform", None)


# ---- applying --------------------------------------------------------------------

def test_an_added_element_appears_with_its_default_feature_in_the_decision():
    candidate = applied(add_elements=[ADD_SARCASM])
    score = candidate.score("Sentiment")

    assert [e.key for e in score.elements] == ["praise", "criticism", "sarcasm"]
    assert score.decision.features[-1] == "sarcasm.logit_p"


def test_the_added_element_is_one_more_question_in_the_same_single_request():
    candidate = applied(add_elements=[ADD_SARCASM])

    assert "sentiment.sarcasm" in candidate.questions()
    assert len(candidate.questions()) == 4


def test_an_agent_supplied_feature_list_overrides_the_default():
    candidate = applied(add_elements=[{**ADD_SARCASM, "features": ["sarcasm.p", "sarcasm.is_yes"]}])

    assert candidate.score("Sentiment").decision.features[-2:] == ["sarcasm.p", "sarcasm.is_yes"]


def test_retiring_an_element_removes_its_question_and_its_features():
    candidate = applied(retire_elements=["criticism"])
    score = candidate.score("Sentiment")

    assert [e.key for e in score.elements] == ["praise"]
    assert "criticism.logit_p" not in score.decision.features
    assert "sentiment.criticism" not in candidate.questions()


def test_rewording_changes_only_the_instructions_and_keeps_the_features():
    candidate = applied(reword_elements=[{"key": "praise", "instructions": "Sincere praise only?"}])
    score = candidate.score("Sentiment")

    assert {e.key: e.instructions for e in score.elements}["praise"] == "Sincere praise only?"
    assert score.decision.features == ["self.holistic.clr.positive", "praise.logit_p",
                                       "criticism.logit_p"]


def test_the_holistic_question_can_be_reworded_but_never_retired():
    candidate = applied(reword_elements=[{"key": "holistic", "instructions": "Real sentiment?"}])
    assert candidate.score("Sentiment").instructions == "Real sentiment?"

    with pytest.raises(ProposalError, match="holistic"):
        applied(retire_elements=["holistic"])


def test_the_heads_numbers_are_cleared_because_they_belong_to_the_old_feature_set():
    # Weights, calibration and provenance are set only by the deterministic fit.
    decision = applied(add_elements=[ADD_SARCASM]).score("Sentiment").decision

    assert decision.weights == {}
    assert decision.calibration is None and decision.provenance is None
    assert decision.model == "multinomial_logistic"


def test_a_hand_written_threshold_rule_becomes_a_fittable_head():
    # The seed scorecard's decision is a hand-written linear_threshold rule. A proposal
    # turns it into a multinomial head with no weights, ready for the fit to fill in.
    from pathlib import Path

    seed = Scorecard.from_yaml(
        (Path(__file__).resolve().parents[1] / "fixtures" / "scorecards" / "v1.yaml").read_text())
    assert seed.score("Sentiment").decision.model == "linear_threshold"

    candidate = apply_proposal(seed, "Sentiment", parse(add_elements=[ADD_SARCASM]))

    decision = candidate.score("Sentiment").decision
    assert decision.model == "multinomial_logistic"
    assert decision.positive_class is None
    assert decision.features == ["self.holistic.clr.positive", "sarcasm.logit_p"]


def test_the_original_scorecard_is_never_modified():
    original = card()
    before = original.to_config()

    apply_proposal(original, "Sentiment", parse(add_elements=[ADD_SARCASM], retire_elements=["praise"]))

    assert original.to_config() == before


def test_a_no_op_proposal_changes_only_the_cleared_numbers():
    candidate = applied()

    assert [e.to_config() for e in candidate.score("Sentiment").elements] == \
           [e.to_config() for e in card().score("Sentiment").elements]


def test_adding_an_element_that_already_exists_says_to_reword_it_instead():
    with pytest.raises(ProposalError, match="reword"):
        applied(add_elements=[{**ADD_SARCASM, "key": "praise"}])


def test_retiring_or_rewording_an_element_that_does_not_exist_lists_the_real_ones():
    with pytest.raises(ProposalError, match="Current elements"):
        applied(retire_elements=["ghost"])
    with pytest.raises(ProposalError, match="Current elements"):
        applied(reword_elements=[{"key": "ghost", "instructions": "x"}])


def test_a_dotted_or_reserved_element_key_is_rejected_by_the_scorecard_rules():
    with pytest.raises(ProposalError, match="not valid"):
        applied(add_elements=[{**ADD_SARCASM, "key": "bad.key"}])
    with pytest.raises(ProposalError, match="not valid"):
        applied(add_elements=[{**ADD_SARCASM, "key": "shared"}])


def test_a_feature_the_element_cannot_produce_is_rejected():
    with pytest.raises(ProposalError, match="not valid"):
        applied(add_elements=[{**ADD_SARCASM, "features": ["sarcasm.clr.maybe"]}])


def test_a_batch_of_changes_lands_as_one_scorecard():
    candidate = applied(
        add_elements=[ADD_SARCASM, {**ADD_SARCASM, "key": "irony", "instructions": "Ironic?"}],
        retire_elements=["criticism"],
        reword_elements=[{"key": "praise", "instructions": "Sincere praise?"}])

    assert [e.key for e in candidate.score("Sentiment").elements] == ["praise", "sarcasm", "irony"]


# ---- diffing ---------------------------------------------------------------------

def test_the_diff_reports_exactly_what_changed_for_the_human_to_approve():
    after = applied(add_elements=[ADD_SARCASM], retire_elements=["criticism"],
                    reword_elements=[{"key": "praise", "instructions": "Sincere praise?"}])

    diff = diff_summary(card(), after, "Sentiment")

    assert diff["added"] == ["sarcasm"]
    assert diff["retired"] == ["criticism"]
    assert diff["reworded"] == ["praise"]
    assert diff["features_added"] == ["sarcasm.logit_p"]
    assert diff["features_removed"] == ["criticism.logit_p"]
    assert diff["holistic_reworded"] is False


def test_identical_scorecards_have_an_empty_diff():
    diff = diff_summary(card(), card(), "Sentiment")

    assert not any(diff[k] for k in ("added", "retired", "reworded", "features_added",
                                     "features_removed", "holistic_reworded"))


def test_a_proposal_object_reports_its_own_summary():
    assert Proposal(root_cause="x").is_noop
