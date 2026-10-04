"""The live JEVPA split is fresh relative to discovery and the prior pilot."""
from scripts.prepare_jevpa_live import split_manifest


def test_live_manifest_freezes_the_registered_fresh_split():
    report = split_manifest("fixtures", "fixtures/recordings/simulated-labeler")

    discovery = set(report["discovery_ids"])
    pilot = (set(report["excluded_prior_pilot"]["selection_ids"])
             | set(report["excluded_prior_pilot"]["test_ids"]))
    selection, test = set(report["selection_ids"]), set(report["test_ids"])

    assert report["split_seed"] == 20260925
    assert len(selection) == 300
    assert len(test) == 500
    assert not selection & test
    assert not (selection | test) & (discovery | pilot)
    assert len(report["sha256"]) == 64
