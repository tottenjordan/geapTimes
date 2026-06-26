"""Offline tests for the minimal point-metric helper."""

import pandas as pd
import pytest

from geaptimes.experiment.metrics import point_metrics
from geaptimes.models.base import (
    DATE_COLUMN,
    FORECAST_COLUMN,
    MODEL_COLUMN,
    SERIES_COLUMN,
)

SERIES_COL = "start_station_name"
TIME_COL = "date"
TARGET_COL = "num_trips"


def _preds(series: list[str], dates: list[str], forecast: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_COLUMN: series,
            DATE_COLUMN: [d.date() for d in pd.to_datetime(dates)],
            FORECAST_COLUMN: forecast,
            MODEL_COLUMN: "m",
        }
    )


def _actuals(series: list[str], dates: list[str], target: list[float]) -> pd.DataFrame:
    return pd.DataFrame({SERIES_COL: series, TIME_COL: pd.to_datetime(dates), TARGET_COL: target})


def _metrics(preds: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, float]:
    return point_metrics(
        preds, actuals, series_col=SERIES_COL, time_col=TIME_COL, target_col=TARGET_COL
    )


def test_exact_mae_and_rmse() -> None:
    preds = _preds(["s", "s"], ["2018-05-18", "2018-05-19"], [10.0, 12.0])
    actuals = _actuals(["s", "s"], ["2018-05-18", "2018-05-19"], [12.0, 12.0])
    # errors: -2, 0  -> mae = 1.0, rmse = sqrt(2)
    out = _metrics(preds, actuals)
    assert out["mae"] == 1.0
    assert out["rmse"] == pytest.approx(2**0.5)
    assert out["n_points"] == 2.0


def test_inner_join_counts_only_overlap() -> None:
    preds = _preds(["s", "s"], ["2018-05-18", "2018-05-19"], [10.0, 10.0])
    # actuals only cover one of the forecast dates
    actuals = _actuals(["s"], ["2018-05-18"], [10.0])
    out = _metrics(preds, actuals)
    assert out["n_points"] == 1.0
    assert out["mae"] == 0.0


def test_empty_overlap_raises() -> None:
    preds = _preds(["s"], ["2018-05-18"], [10.0])
    actuals = _actuals(["s"], ["2019-01-01"], [10.0])
    with pytest.raises(ValueError, match="no overlapping"):
        _metrics(preds, actuals)
