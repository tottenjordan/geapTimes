"""Offline tests for the point-metric helper and the Stage 5 standardized eval suite."""

from typing import cast

import pandas as pd
import pytest

from geaptimes.experiment.metrics import evaluate, point_metrics
from geaptimes.models.base import (
    DATE_COLUMN,
    FORECAST_COLUMN,
    MODEL_COLUMN,
    QUANTILES,
    SERIES_COLUMN,
    quantile_column,
)

SERIES_COL = "start_station_name"
TIME_COL = "date"
TARGET_COL = "num_trips"


def _preds(
    series: list[str],
    dates: list[str],
    forecast: list[float],
    quantiles: dict[float, list[float]] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            SERIES_COLUMN: series,
            DATE_COLUMN: [d.date() for d in pd.to_datetime(dates)],
            FORECAST_COLUMN: forecast,
            MODEL_COLUMN: "m",
        }
    )
    # Default: every qXX column mirrors the point forecast (so pinball loss == 0).
    for q in QUANTILES:
        col = quantile_column(q)
        frame[col] = (quantiles or {}).get(q, forecast)
    return frame


def _actuals(series: list[str], dates: list[str], target: list[float]) -> pd.DataFrame:
    return pd.DataFrame({SERIES_COL: series, TIME_COL: pd.to_datetime(dates), TARGET_COL: target})


def _metrics(preds: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, float]:
    return point_metrics(
        preds, actuals, series_col=SERIES_COL, time_col=TIME_COL, target_col=TARGET_COL
    )


def _evaluate(preds: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, object]:
    return evaluate(preds, actuals, series_col=SERIES_COL, time_col=TIME_COL, target_col=TARGET_COL)


# --- point_metrics (Stage 3, retained) -------------------------------------------------------


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


# --- evaluate (Stage 5 suite) --------------------------------------------------------------


def test_evaluate_superset_matches_point_metrics() -> None:
    preds = _preds(["s", "s"], ["2018-05-18", "2018-05-19"], [10.0, 12.0])
    actuals = _actuals(["s", "s"], ["2018-05-18", "2018-05-19"], [12.0, 12.0])
    point = _metrics(preds, actuals)
    full = _evaluate(preds, actuals)
    # mae / rmse / n_points are byte-for-byte the same as the minimal helper.
    assert full["mae"] == point["mae"]
    assert full["rmse"] == point["rmse"]
    assert full["n_points"] == point["n_points"]


def test_evaluate_smape_exact() -> None:
    # forecast 10 vs actual 12: |error|=2, denom=(12+10)/2=11 -> 2/11; second point perfect -> 0.
    preds = _preds(["s", "s"], ["2018-05-18", "2018-05-19"], [10.0, 12.0])
    actuals = _actuals(["s", "s"], ["2018-05-18", "2018-05-19"], [12.0, 12.0])
    out = _evaluate(preds, actuals)
    expected = ((2.0 / 11.0) + 0.0) / 2.0 * 100.0
    assert out["smape"] == pytest.approx(expected)


def test_evaluate_smape_guards_both_zero() -> None:
    # actual == forecast == 0 is a perfect prediction: contributes 0, not NaN.
    preds = _preds(["s"], ["2018-05-18"], [0.0])
    actuals = _actuals(["s"], ["2018-05-18"], [0.0])
    out = _evaluate(preds, actuals)
    assert out["smape"] == 0.0
    assert out["mae"] == 0.0


def test_evaluate_quantile_loss_zero_when_quantiles_match_actual() -> None:
    # Every qXX == actual -> pinball loss is exactly 0 across all quantiles.
    actual = [12.0, 12.0]
    preds = _preds(
        ["s", "s"],
        ["2018-05-18", "2018-05-19"],
        [12.0, 12.0],
        quantiles=dict.fromkeys(QUANTILES, actual),
    )
    actuals = _actuals(["s", "s"], ["2018-05-18", "2018-05-19"], actual)
    out = _evaluate(preds, actuals)
    assert out["quantile_loss"] == pytest.approx(0.0)


def test_evaluate_quantile_loss_exact() -> None:
    # Single point, actual=10. Each qXX over/under-predicts; pinball averaged across quantiles.
    actual = 10.0
    qpreds = {0.1: [8.0], 0.3: [9.0], 0.5: [10.0], 0.7: [11.0], 0.9: [13.0]}
    preds = _preds(["s"], ["2018-05-18"], [10.0], quantiles=qpreds)
    actuals = _actuals(["s"], ["2018-05-18"], [actual])
    out = _evaluate(preds, actuals)

    def pinball(q: float, pred: float) -> float:
        delta = actual - pred
        return max(q * delta, (q - 1.0) * delta)

    losses = [pinball(q, qpreds[q][0]) for q in QUANTILES]
    assert out["quantile_loss"] == pytest.approx(sum(losses) / len(losses))


def test_evaluate_per_series_breakdown() -> None:
    preds = _preds(
        ["a", "a", "b"],
        ["2018-05-18", "2018-05-19", "2018-05-18"],
        [10.0, 10.0, 20.0],
    )
    actuals = _actuals(
        ["a", "a", "b"],
        ["2018-05-18", "2018-05-19", "2018-05-18"],
        [12.0, 8.0, 20.0],
    )
    out = _evaluate(preds, actuals)
    rows = cast("list[dict[str, object]]", out["per_series"])
    per_series = {row["series"]: row for row in rows}
    assert set(per_series) == {"a", "b"}
    # series "a": errors -2, +2 -> mae 2.0; series "b": perfect -> mae 0.0
    assert per_series["a"]["mae"] == pytest.approx(2.0)
    assert per_series["b"]["mae"] == 0.0
    assert per_series["a"]["n_points"] == 2.0
    assert per_series["b"]["n_points"] == 1.0


def test_evaluate_empty_overlap_raises() -> None:
    preds = _preds(["s"], ["2018-05-18"], [10.0])
    actuals = _actuals(["s"], ["2019-01-01"], [10.0])
    with pytest.raises(ValueError, match="no overlapping"):
        _evaluate(preds, actuals)
