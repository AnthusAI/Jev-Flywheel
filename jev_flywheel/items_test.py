"""Feature: items, feedback, and label normalization.

Normalization has to match Plexus exactly. A mismatch does not raise; it
silently sends every metric to zero, which is why it is spec'd this closely.
"""
import pytest

from jev_flywheel.items import (
    LABEL_SOURCE_FINAL,
    LABEL_SOURCE_SCORE_RESULT_OR_IMPORTED,
    LABEL_SOURCE_VETTED,
    FeedbackItem,
    Item,
    JsonlStore,
    agrees,
    normalize_label,
    normalize_prediction,
)


def test_predictions_are_lowercased_and_whitespace_collapsed():
    assert normalize_prediction("  Positive  ") == "positive"
    assert normalize_prediction("Fell   Short") == "fell short"


def test_labels_lose_trailing_sentence_punctuation():
    assert normalize_label("Positive.") == "positive"
    assert normalize_label("Yes!") == "yes"
    assert normalize_label("Really?") == "really"


def test_the_several_spellings_of_missing_collapse():
    assert normalize_label("nan") == ""
    assert normalize_label(None) == ""
    assert normalize_label("N/A") == "na"


def test_agreement_is_exact_after_normalization():
    assert agrees("Positive", "positive.")
    assert agrees("positive", "Positive")
    assert not agrees("positive", "negative")


def test_every_spelling_of_missing_agrees_with_every_other():
    # Plexus collapses '', 'nan', 'none', 'null' and 'n/a' onto 'na' before
    # comparing, so a blank prediction counts as correct against a blank label.
    assert agrees("", None)
    assert agrees("none", "nan")
    assert not agrees("", "positive")


def test_an_item_exposes_its_split_and_reference_label_from_metadata():
    item = Item(id="a", text="x", metadata={"split": "test", "reference_label": "positive"})
    assert item.split == "test"
    assert item.reference_label == "positive"


def test_an_item_without_corpus_metadata_has_neither():
    item = Item(id="a", text="x")
    assert item.split is None
    assert item.reference_label is None


def test_a_trusted_feedback_item_yields_its_final_answer_as_the_label():
    feedback = FeedbackItem(
        id="f1", item_id="a", score_name="Sentiment",
        initial_answer_value="negative", final_answer_value="positive",
        label_source=LABEL_SOURCE_VETTED)
    assert feedback.label == "positive"


def test_a_label_from_the_ais_own_prediction_is_not_trainable():
    # score_result_or_imported falls back to initialAnswerValue, which is the
    # incumbent's own guess. Training on it teaches the head to imitate the
    # champion, and it looks like it is working because it agrees with the
    # baseline.
    feedback = FeedbackItem(
        id="f1", item_id="a", score_name="Sentiment", final_answer_value="positive",
        label_source=LABEL_SOURCE_SCORE_RESULT_OR_IMPORTED)
    assert feedback.label is None


def test_an_invalidated_feedback_item_yields_no_label():
    feedback = FeedbackItem(
        id="f1", item_id="a", score_name="Sentiment", final_answer_value="positive",
        label_source=LABEL_SOURCE_VETTED, metadata={"is_invalid": True})
    assert feedback.label is None


def test_feedback_records_the_confusion_cell_it_lands_in():
    feedback = FeedbackItem(
        id="f1", item_id="a", score_name="Sentiment",
        initial_answer_value="Negative", final_answer_value="positive.")
    assert feedback.confusion_cell == "negative->positive"


def test_feedback_exposes_the_selection_propensity_that_produced_it():
    feedback = FeedbackItem(id="f1", item_id="a", score_name="S", metadata={"propensity": 0.02})
    assert feedback.propensity == pytest.approx(0.02)
    assert FeedbackItem(id="f2", item_id="b", score_name="S").propensity is None


def test_the_store_round_trips_records(tmp_path):
    store = JsonlStore(tmp_path / "feedback.jsonl", FeedbackItem)
    store.append(FeedbackItem(id="f1", item_id="a", score_name="S", final_answer_value="positive"))
    store.append(FeedbackItem(id="f2", item_id="b", score_name="S", final_answer_value="negative"))

    loaded = store.all()

    assert [f.id for f in loaded] == ["f1", "f2"]
    assert loaded[0].final_answer_value == "positive"


def test_a_store_that_does_not_exist_yet_reads_as_empty(tmp_path):
    assert JsonlStore(tmp_path / "missing.jsonl", FeedbackItem).all() == []


def test_a_later_record_supersedes_an_earlier_one_for_the_same_key(tmp_path):
    # An append-only log expresses a correction by appending, so a human
    # revisiting an item must not leave two live labels behind.
    store = JsonlStore(tmp_path / "feedback.jsonl", FeedbackItem)
    store.append(FeedbackItem(id="f1", item_id="a", score_name="S", final_answer_value="positive"))
    store.append(FeedbackItem(id="f2", item_id="a", score_name="S", final_answer_value="negative"))

    latest = store.latest_by("item_id")

    assert len(latest) == 1
    assert latest["a"].final_answer_value == "negative"


def test_stored_rows_with_unknown_fields_still_load(tmp_path):
    # Fixtures are committed, so an older file has to survive a new field.
    path = tmp_path / "feedback.jsonl"
    path.write_text('{"id": "f1", "item_id": "a", "score_name": "S", "from_the_future": 1}\n')

    loaded = JsonlStore(path, FeedbackItem).all()

    assert loaded[0].id == "f1"
