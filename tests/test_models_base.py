"""Tests for the Forecaster ABC and shared prediction contract."""

import pandas as pd
import pytest

from geaptimes.models.base import (
    PREDICTION_COLUMNS,
    QUANTILE_COLUMNS,
    QUANTILES,
    Forecaster,
    ForecastResult,
    quantile_column,
)
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "p", "gcs_bucket": "b"},
    "models": [{"name": "t", "params": {"type": "timesfm"}}],
    "execution": {"experiment_name": "exp"},
}


def test_quantiles_and_columns() -> None:
    assert QUANTILES == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert QUANTILE_COLUMNS == ["q10", "q30", "q50", "q70", "q90"]
    assert quantile_column(0.5) == "q50"
    assert PREDICTION_COLUMNS[0] == "series"
    assert PREDICTION_COLUMNS[-1] == "model"
    assert set(QUANTILE_COLUMNS).issubset(PREDICTION_COLUMNS)


def test_abc_cannot_be_instantiated() -> None:
    cfg = ExperimentConfig.model_validate(BASE)
    with pytest.raises(TypeError):
        Forecaster(cfg.models[0], cfg)  # type: ignore[abstract]


def test_trivial_subclass_satisfies_interface() -> None:
    cfg = ExperimentConfig.model_validate(BASE)

    class Dummy(Forecaster):
        def fit(self) -> None:
            return None

        def predict(self) -> ForecastResult:
            return ForecastResult(predictions=pd.DataFrame(columns=PREDICTION_COLUMNS))

    model = Dummy(cfg.models[0], cfg)
    assert model.name == "t"
    model.fit()
    result = model.predict()
    assert list(result.predictions.columns) == PREDICTION_COLUMNS
    assert result.metadata == {}
