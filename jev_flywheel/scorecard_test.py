"""Feature: the whole-scorecard YAML.

One file describes every question in one Jev request. It is also what the
steering procedure rewrites, so it has to round-trip exactly and reject a
malformed candidate at load time with a message naming the problem.
"""
import math

import pytest
import yaml

from jev_flywheel.scorecard import ConfigError, Scorecard, slug

SENTIMENT = """
name: Sentiment
version: 2
scores:
  - name: Sentiment
    key: sentiment
    question_type: choice
    instructions: What is the overall sentiment of this text?
    criteria: {positive: null, negative: null}
    elements:
      - key: praise
        question_type: noul
        instructions: Does the text express praise?
      - key: intensity
        question_type: score
        instructions: How strong is the emotion?
        criteria: [none, mild, moderate, strong]
    decision:
      model: multinomial_logistic
      classes: [positive, negative]
      features:
        - self.holistic.clr.positive
        - praise.logit_p
        - intensity.expected_level
      parameters:
        weights:
          positive:
            intercept: 0.3
            self.holistic.clr.positive: 1.8
            praise.logit_p: 1.2
            intensity.expected_level: -0.5
"""


def load(text=SENTIMENT):
    return Scorecard.from_yaml(text)


def test_a_slug_is_wire_safe():
    assert slug("Objection Handled") == "objection_handled"
    assert slug("Agent  Mis.Representation") == "agent_mis_representation"


def test_a_scorecard_loads_its_scores_elements_and_decision():
    card = load()

    assert card.name == "Sentiment"
    assert card.version == 2
    score = card.score("Sentiment")
    assert [e.key for e in score.elements] == ["praise", "intensity"]
    assert score.decision.model == "multinomial_logistic"
    assert score.decision.weights["positive"]["praise.logit_p"] == 1.2


def test_the_whole_card_becomes_one_question_set_with_namespaced_elements():
    # This is the economic heart of the design: every question below rides in a
    # single request whose cost is dominated by the item's text.
    questions = load().questions()

    assert set(questions) == {"Sentiment", "sentiment.praise", "sentiment.intensity"}
    assert questions["Sentiment"]["type"] == "choice"
    assert questions["sentiment.praise"]["type"] == "noul"


def test_a_noul_question_carries_no_criteria_key_when_it_has_none():
    # A null criteria passes the SDK's presence check and then fails at the API.
    assert "criteria" not in load().questions()["sentiment.praise"]


def test_features_are_extracted_from_the_shared_answer_set():
    card = load()
    answers = {
        "Sentiment": {"type": "choice", "choice": "positive",
                      "probabilities": {"positive": 0.8, "negative": 0.2}},
        "sentiment.praise": {"type": "noul", "noul": 0.9},
        "sentiment.intensity": {"type": "score", "score": 3.0,
                                "probabilities": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}},
    }

    vector = card.score("Sentiment").feature_vector(answers)

    assert set(vector) == set(card.score("Sentiment").decision.features)
    assert vector["praise.logit_p"] == pytest.approx(math.log(0.9 / 0.1))
    assert vector["intensity.expected_level"] == pytest.approx(1.0)


def test_an_unanswered_element_is_omitted_rather_than_zero_filled():
    # Omitting lets serving report coverage and lets fitting refuse. Silently
    # zero-filling would hide both.
    card = load()
    answers = {"Sentiment": {"choice": "positive",
                             "probabilities": {"positive": 0.8, "negative": 0.2}},
               "sentiment.praise": {"noul": 0.9}}

    vector = card.score("Sentiment").feature_vector(answers)

    assert "intensity.expected_level" not in vector
    assert "praise.logit_p" in vector


def test_a_scorecard_round_trips_through_yaml():
    # The steering procedure rewrites this file, so a lossy round trip would
    # quietly drop configuration between versions.
    original = load()

    reloaded = Scorecard.from_yaml(original.to_yaml())

    assert reloaded.to_config() == original.to_config()


def test_a_decision_referencing_an_undeclared_element_is_rejected():
    bad = SENTIMENT.replace("praise.logit_p", "nonexistent.logit_p")
    with pytest.raises(ConfigError, match="nonexistent"):
        load(bad)


def test_a_decision_asking_for_a_term_the_element_cannot_produce_is_rejected():
    bad = SENTIMENT.replace("praise.logit_p", "praise.clr.maybe")
    with pytest.raises(ConfigError, match="clr.maybe"):
        load(bad)


def test_an_element_key_with_a_dot_is_rejected():
    # Feature names split on dots, so a dotted key would make the split ambiguous.
    bad = SENTIMENT.replace("key: praise", "key: pr.aise")
    with pytest.raises(ConfigError, match=r"pr\.aise"):
        load(bad)


@pytest.mark.parametrize("reserved", ["self", "shared"])
def test_a_reserved_element_key_is_rejected(reserved):
    bad = SENTIMENT.replace("key: praise", f"key: {reserved}")
    with pytest.raises(ConfigError, match="reserved"):
        load(bad)


def test_a_choice_element_without_criteria_is_rejected():
    bad = SENTIMENT.replace(
        "      - key: praise\n        question_type: noul\n"
        "        instructions: Does the text express praise?\n",
        "      - key: praise\n        question_type: choice\n"
        "        instructions: Does the text express praise?\n")
    with pytest.raises(ConfigError, match="criteria"):
        load(bad)


def test_a_score_with_neither_a_question_nor_a_decision_is_rejected():
    with pytest.raises(ConfigError, match="question_type"):
        Scorecard.from_yaml("name: X\nscores:\n  - name: Bare\n")


def test_a_duplicate_element_key_is_rejected():
    bad = SENTIMENT.replace("key: intensity", "key: praise")
    with pytest.raises(ConfigError, match="duplicate"):
        load(bad)


def test_duplicate_score_names_are_rejected():
    bad = SENTIMENT + """
  - name: Sentiment
    question_type: noul
    instructions: Again?
"""
    with pytest.raises(ConfigError, match="duplicate score"):
        load(bad)


def test_two_scores_may_share_an_element_when_the_definitions_agree():
    shared = """
name: Card
scores:
  - name: A
    key: a
    question_type: noul
    instructions: A?
    shared_elements:
      - {key: transferred, question_type: noul, instructions: "Was it transferred?"}
  - name: B
    key: b
    question_type: noul
    instructions: B?
    shared_elements:
      - {key: transferred, question_type: noul, instructions: "Was it transferred?"}
"""
    questions = Scorecard.from_yaml(shared).questions()

    # One registration, asked once, used by both scores.
    assert set(questions) == {"A", "B", "shared.transferred"}


def test_a_shared_element_defined_two_different_ways_is_an_authoring_error():
    conflicting = """
name: Card
scores:
  - name: A
    key: a
    question_type: noul
    instructions: A?
    shared_elements:
      - {key: transferred, question_type: noul, instructions: one}
  - name: B
    key: b
    question_type: noul
    instructions: B?
    shared_elements:
      - {key: transferred, question_type: noul, instructions: two}
"""
    with pytest.raises(ConfigError, match="shared.transferred"):
        Scorecard.from_yaml(conflicting)


def test_a_hand_authored_threshold_rule_needs_no_weights_to_be_fitted():
    # A stakeholder can write a rule before any labels exist.
    card = Scorecard.from_yaml("""
name: Card
scores:
  - name: Escalate
    key: escalate
    question_type: noul
    instructions: Should this escalate?
    elements:
      - {key: angry, question_type: noul, instructions: "Is the customer angry?"}
    decision:
      model: linear_threshold
      classes: ["Yes", "No"]
      positive_class: "Yes"
      threshold: 0.5
      features: [angry.p]
      parameters:
        weights: {angry.p: 1.0}
""")
    assert card.score("Escalate").decision.model == "linear_threshold"
    assert card.score("Escalate").decision.classes == ["Yes", "No"]


def test_unquoted_yes_and_no_class_labels_are_rejected():
    # YAML 1.1 reads bare Yes/No as booleans, so these arrive as [True, False]
    # and would become the strings "True"/"False". Nothing would raise; every
    # metric would silently go to zero. The steering agent writes this file, so
    # the trap has to be caught here.
    with pytest.raises(ConfigError, match="unquoted"):
        Scorecard.from_yaml("""
name: Card
scores:
  - name: Escalate
    key: escalate
    question_type: noul
    instructions: "Should this escalate?"
    decision:
      model: multinomial_logistic
      classes: [Yes, No]
      features: [self.holistic.logit_p]
      parameters:
        weights:
          Yes: {intercept: 0.0, self.holistic.logit_p: 1.0}
""")


def test_unquoted_yes_and_no_criteria_options_are_rejected():
    with pytest.raises(ConfigError, match="unquoted"):
        Scorecard.from_yaml("""
name: Card
scores:
  - name: Agreed
    key: agreed
    question_type: choice
    instructions: "Did they agree?"
    criteria: {Yes: null, No: null}
""")


def test_malformed_yaml_is_reported_as_a_config_error():
    with pytest.raises(ConfigError, match="YAML"):
        Scorecard.from_yaml("name: [unclosed\n")


def test_a_scorecard_needs_scores():
    with pytest.raises(ConfigError, match="scores"):
        Scorecard.from_yaml("name: Empty\n")
