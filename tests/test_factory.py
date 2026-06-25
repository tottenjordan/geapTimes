"""Tests for ForecastFactory dispatch (no model loads, no cloud)."""

import pytest

from geaptimes.models.automl import AutoMLForecaster
from geaptimes.models.bqml import BQMLForecaster
from geaptimes.models.factory import ForecastFactory
from geaptimes.models.timesfm import TimesFMForecaster
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "p", "gcs_bucket": "b"},
    "execution": {"experiment_name": "exp"},
}


def _cfg(models: list[dict]) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, "models": models})


def test_create_dispatches_to_each_backend() -> None:
    timesfm = _cfg([{"name": "t", "params": {"type": "timesfm"}}])
    bqml = _cfg([{"name": "b", "params": {"type": "bqml"}}])
    automl = _cfg([{"name": "a", "params": {"type": "automl"}}])

    assert isinstance(ForecastFactory.create(timesfm.models[0], timesfm), TimesFMForecaster)
    assert isinstance(ForecastFactory.create(bqml.models[0], bqml), BQMLForecaster)
    assert isinstance(ForecastFactory.create(automl.models[0], automl), AutoMLForecaster)


def test_from_config_skips_disabled() -> None:
    cfg = _cfg(
        [
            {"name": "t", "params": {"type": "timesfm"}},
            {"name": "b", "enabled": False, "params": {"type": "bqml"}},
            {"name": "a", "params": {"type": "automl"}},
        ]
    )
    forecasters = ForecastFactory.from_config(cfg)
    assert [f.name for f in forecasters] == ["t", "a"]  # disabled bqml skipped


def test_from_config_all_disabled_returns_empty() -> None:
    cfg = _cfg([{"name": "t", "enabled": False, "params": {"type": "timesfm"}}])
    assert ForecastFactory.from_config(cfg) == []


def test_unknown_backend_raises() -> None:
    cfg = _cfg([{"name": "t", "params": {"type": "timesfm"}}])
    model_cfg = cfg.models[0]
    # Force an unsupported discriminator past schema validation.
    object.__setattr__(model_cfg.params, "type", "mystery")
    with pytest.raises(ValueError, match="Unknown forecaster backend"):
        ForecastFactory.create(model_cfg, cfg)
