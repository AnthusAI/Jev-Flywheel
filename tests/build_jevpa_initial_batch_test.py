"""Initial analyst key collisions are made safe without changing question wording."""
import json

from scripts.build_jevpa_initial_batch import build


def test_initial_batch_disambiguates_reused_keys(tmp_path):
    proposals = tmp_path / "proposals.jsonl"
    reply = lambda key, question: json.dumps({"add_elements": [
        {"key": key, "question_type": "noul", "instructions": question}]})
    proposals.write_text("\n".join(json.dumps({"call_index": index, "analyst_reply": reply(key, text)})
                                   for index, key, text in [
        (1, "topic", "Is this about sport?"), (2, "topic", "Is this about work?"),
        (3, "form", "Is this formal?"), (4, "length", "Is this long?")]) + "\n")

    report = build(proposals, "fixtures/scorecards/v1.yaml", tmp_path / "cards")

    assert [row["question_key"] for row in report["candidates"]] == [
        "topic", "topic_2", "form", "length"]
    assert report["question_count"] == 5
