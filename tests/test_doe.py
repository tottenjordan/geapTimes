"""Offline tests for the DOE matrix engine (pure config expansion; no cloud)."""

import pytest
from pydantic import ValidationError

from geaptimes.experiment.doe import DOEPoint, expand, point_slug
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "p", "gcs_bucket": "b"},
    "models": [{"name": "timesfm", "params": {"type": "timesfm"}}],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def test_empty_axes_yields_single_base_point() -> None:
    points = expand(_cfg())
    assert len(points) == 1
    assert points[0].overrides == {}
    assert points[0].config.forecast.horizon == 14


def test_single_axis_expands_per_value() -> None:
    points = expand(_cfg(doe={"axes": {"forecast.horizon": [7, 14]}}))
    assert [p.config.forecast.horizon for p in points] == [7, 14]
    assert [p.overrides for p in points] == [
        {"forecast.horizon": 7},
        {"forecast.horizon": 14},
    ]


def test_cross_product_size_and_overrides() -> None:
    points = expand(
        _cfg(doe={"axes": {"forecast.horizon": [7, 14], "data.station_filter.top_n": [10, 25]}})
    )
    assert len(points) == 4  # 2 x 2
    combos = {(p.config.forecast.horizon, p.config.data.station_filter.top_n) for p in points}
    assert combos == {(7, 10), (7, 25), (14, 10), (14, 25)}


def test_override_is_revalidated() -> None:
    with pytest.raises(ValidationError, match="horizon"):
        expand(_cfg(doe={"axes": {"forecast.horizon": [0]}}))


def test_unknown_axis_path_raises() -> None:
    with pytest.raises(ValueError, match="does not resolve"):
        expand(_cfg(doe={"axes": {"forecast.nope": [1]}}))


def test_base_config_not_mutated() -> None:
    cfg = _cfg(doe={"axes": {"forecast.horizon": [7]}})
    expand(cfg)
    assert cfg.forecast.horizon == 14  # original untouched


def test_variant_drops_doe_block() -> None:
    points = expand(_cfg(doe={"axes": {"forecast.horizon": [7]}}))
    assert points[0].config.doe.axes == {}


def test_point_slug() -> None:
    assert point_slug({}) == "base"
    assert point_slug({"forecast.horizon": 14}) == "horizon14"
    assert (
        point_slug({"forecast.horizon": 7, "data.station_filter.top_n": 10}) == "horizon7-top_n10"
    )


def test_doe_point_dataclass_defaults() -> None:
    cfg = _cfg()
    assert DOEPoint(config=cfg).overrides == {}
