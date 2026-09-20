"""Feature: the capability ladder gates what a fit may do on effective sample size."""
import pytest

from jev_flywheel.ladder import (
    CAPABILITY_LADDER_V1,
    LADDER_NAME,
    distance_to_next,
    next_tier,
    tier_for,
)
from jev_flywheel.models import REGISTRY


@pytest.mark.parametrize("n_effective, expected", [
    (0, "hold"), (29.9, "hold"),
    (30, "shrunk"), (199, "shrunk"),
    (200, "standard"), (999, "standard"),
    (1000, "rich"), (50000, "rich"),
])
def test_each_effective_sample_size_selects_the_expected_tier(n_effective, expected):
    assert tier_for(n_effective).name == expected


def test_thresholds_increase_up_the_ladder():
    floors = [t.min_n_effective for t in CAPABILITY_LADDER_V1]
    assert floors == sorted(floors) and len(set(floors)) == len(floors)


def test_every_tier_names_a_model_the_registry_can_serve():
    assert all(t.model in REGISTRY for t in CAPABILITY_LADDER_V1)


def test_a_tier_that_fits_a_model_is_gated_by_that_models_own_floor():
    # The ladder must never permit a model below the floor the model itself declares.
    for tier in CAPABILITY_LADDER_V1:
        if tier.name != "hold":
            assert tier.min_n_effective >= REGISTRY[tier.model].min_n_effective * 0.5


def test_the_hold_tier_fits_nothing():
    hold = CAPABILITY_LADDER_V1[0]

    assert hold.c_grid == ()
    assert hold.calibration == "none"
    assert hold.feature_budget(10_000) == 0


def test_calibration_gets_richer_only_as_labels_accumulate():
    # Isotonic memorizes on small samples, so it is reserved for the top rung.
    methods = {t.name: t.calibration for t in CAPABILITY_LADDER_V1}

    assert methods["shrunk"] == "temperature"
    assert methods["rich"] == "two_stage"
    assert "isotonic" not in (methods["shrunk"], methods["standard"])


def test_the_feature_budget_grows_with_evidence_and_shrinks_with_less_shrinkage():
    shrunk = tier_for(60)
    standard = tier_for(400)

    assert shrunk.feature_budget(60) == 12          # heavy shrinkage tolerates more per label
    assert standard.feature_budget(400) == 40
    assert shrunk.feature_budget(100) > shrunk.feature_budget(60)


def test_regularization_options_widen_as_evidence_grows():
    grids = [len(t.c_grid) for t in CAPABILITY_LADDER_V1 if t.name != "hold"]
    assert grids == sorted(grids)


def test_the_next_tier_and_the_distance_to_it():
    assert next_tier(10).name == "shrunk"
    assert distance_to_next(10) == pytest.approx(20)
    assert next_tier(250).name == "rich"
    assert distance_to_next(250) == pytest.approx(750)


def test_the_top_of_the_ladder_has_nowhere_to_go():
    assert next_tier(5000) is None
    assert distance_to_next(5000) is None


def test_the_ladder_is_immutable_so_a_recorded_tier_stays_interpretable():
    # Frozen dataclasses in a tuple: a fit that records "standard under
    # capability-ladder-v1" must mean the same thing next year.
    assert isinstance(CAPABILITY_LADDER_V1, tuple)
    with pytest.raises(Exception):
        CAPABILITY_LADDER_V1[1].min_n_effective = 5
    assert LADDER_NAME == "capability-ladder-v1"
