"""JEVPA branch proposals share the discovery labels and prohibit Jev spend."""
from pathlib import Path

from jev_flywheel.steer import SteerOutcome
from scripts.collect_jevpa_branch_proposals import collect


def test_branch_calls_use_two_slots_per_arm_without_spend(tmp_path):
    calls = []

    def fake_round(workspace, score_name, **kwargs):
        calls.append((workspace.n_labeled(score_name), kwargs["allow_spend"]))
        return SteerOutcome("needs_spend", analyst_reply='{"add_elements": []}')

    cache = Path("var/jevpa_live/answer_workspace/answers.jsonl")
    result = collect(tmp_path / "branches.jsonl", tmp_path / "workspaces",
                     Path("fixtures/recordings/simulated-labeler"), cache,
                     Path("var/jevpa_live/initial/initial-2-fitted.yaml"),
                     Path("var/jevpa_live/initial/initial-4-fitted.yaml"), run_round=fake_round)

    assert [row["phase"] for row in result] == ["greedy", "greedy", "pareto", "pareto"]
    assert calls == [(140, False)] * 4
