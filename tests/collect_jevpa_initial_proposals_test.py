"""JEVPA's initial analyst bank is repeated against discovery data without Jev spend."""
from pathlib import Path

from jev_flywheel.steer import SteerOutcome
from scripts.collect_jevpa_initial_proposals import collect


def test_initial_bank_uses_identical_discovery_size_and_disables_spend(tmp_path):
    calls = []

    def fake_round(workspace, score_name, **kwargs):
        calls.append((workspace.n_labeled(score_name), kwargs["allow_spend"], kwargs["max_auto_requests"]))
        return SteerOutcome("needs_spend", {"requests": 140}, analyst_reply='{"add_elements": []}')

    records = collect(tmp_path / "initial.jsonl", tmp_path / "workspaces",
                      Path("fixtures/recordings/simulated-labeler"), n_calls=2,
                      run_round=fake_round)

    assert calls == [(140, False, 10_000), (140, False, 10_000)]
    assert [row["call_index"] for row in records] == [1, 2]
    assert all(row["allow_spend"] is False for row in records)
