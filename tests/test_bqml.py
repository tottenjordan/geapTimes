"""Offline tests for BQMLForecaster (SQL strings + result mapping; no BigQuery)."""

import pandas as pd

from geaptimes.models.base import PREDICTION_COLUMNS
from geaptimes.models.bqml import (
    BQMLForecaster,
    build_create_model_sql,
    build_forecast_sql,
)
from geaptimes.schemas import ExperimentConfig

COVS = ["temp", "prcp", "day_of_week", "month", "is_weekend"]

BASE = {
    "project": {"id": "p", "gcs_bucket": "b", "bq_dataset": "ds"},
    "data": {"covariates": {"available_at_forecast_columns": COVS}},
    "models": [
        {
            "name": "bqml_arima_xreg",
            "params": {"type": "bqml", "model_type": "ARIMA_PLUS_XREG", "use_covariates": True},
        }
    ],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def test_create_model_sql_xreg() -> None:
    cfg = _cfg()  # holiday_region defaults to "US"
    sql = build_create_model_sql(
        cfg,
        cfg.models[0].params,  # type: ignore[arg-type]
        model_id="p.ds.m",
        train_table="p.ds.train",
    )
    assert "CREATE OR REPLACE MODEL `p.ds.m`" in sql
    assert "model_type = 'ARIMA_PLUS_XREG'" in sql
    assert "time_series_timestamp_col = 'date'" in sql
    assert "time_series_data_col = 'num_trips'" in sql
    assert "time_series_id_col = 'start_station_name'" in sql
    assert "data_frequency = 'DAILY'" in sql
    assert "holiday_region = 'US'" in sql
    assert 'labels = [("solution", "geaptimes")]' in sql
    assert "FROM `p.ds.train`" in sql
    for cov in COVS:  # XREG regressors are selected for training
        assert cov in sql


def test_create_model_sql_plain_arima_omits_regressors() -> None:
    cfg = _cfg(
        data={"covariates": {"available_at_forecast_columns": COVS}, "holiday_region": None},
        models=[{"name": "bqml_arima", "params": {"type": "bqml", "model_type": "ARIMA_PLUS"}}],
    )
    sql = build_create_model_sql(
        cfg,
        cfg.models[0].params,  # type: ignore[arg-type]
        model_id="p.ds.m",
        train_table="p.ds.train",
    )
    assert "model_type = 'ARIMA_PLUS'" in sql
    assert "holiday_region" not in sql
    # plain ARIMA selects only id/timestamp/data cols, no weather regressors
    assert "temp" not in sql
    assert "SELECT start_station_name, date, num_trips\n" in sql


def test_forecast_sql_xreg_joins_infer() -> None:
    cfg = _cfg()
    sql = build_forecast_sql(
        cfg,
        cfg.models[0].params,  # type: ignore[arg-type]
        model_id="p.ds.m",
        infer_table="p.ds.infer",
    )
    assert "ML.FORECAST(" in sql
    assert "STRUCT(14 AS horizon, 0.9 AS confidence_level)" in sql
    assert "FROM `p.ds.infer`" in sql
    assert "temp" in sql


def test_forecast_sql_plain_arima_has_no_future_table() -> None:
    cfg = _cfg(
        forecast={"horizon": 7},
        models=[{"name": "bqml_arima", "params": {"type": "bqml", "model_type": "ARIMA_PLUS"}}],
    )
    sql = build_forecast_sql(
        cfg,
        cfg.models[0].params,  # type: ignore[arg-type]
        model_id="p.ds.m",
        infer_table="p.ds.infer",
    )
    assert "ML.FORECAST(" in sql
    assert "infer" not in sql  # no future-regressor table argument
    assert "STRUCT(7 AS horizon" in sql


def test_model_id_includes_config_slug() -> None:
    cfg = _cfg()
    fc = BQMLForecaster(cfg.models[0], cfg, query_runner=lambda _sql: pd.DataFrame())
    assert fc.model_id == "p.ds.bqml_arima_xreg__top25_h14_t14_v14"


def test_fit_runs_create_model() -> None:
    cfg = _cfg()
    seen: list[str] = []

    def runner(sql: str) -> pd.DataFrame:
        seen.append(sql)
        return pd.DataFrame()

    BQMLForecaster(cfg.models[0], cfg, query_runner=runner).fit()
    assert "CREATE OR REPLACE MODEL" in seen[0]


def test_predict_maps_forecast_and_quantiles() -> None:
    cfg = _cfg()
    raw = pd.DataFrame(
        {
            "start_station_name": ["s0", "s0"],
            "forecast_timestamp": pd.to_datetime(["2018-05-18", "2018-05-19"]),
            "forecast_value": [10.0, 12.0],
            "standard_error": [2.0, 2.0],
        }
    )
    result = BQMLForecaster(cfg.models[0], cfg, query_runner=lambda _sql: raw).predict()
    preds = result.predictions
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert result.metadata["model_type"] == "ARIMA_PLUS_XREG"
    first = preds.iloc[0]
    assert first["forecast"] == 10.0
    assert first["q50"] == 10.0  # median == point for ARIMA
    assert abs(first["q90"] - (10.0 + 1.2815515594 * 2.0)) < 1e-6
    assert abs(first["q10"] - (10.0 - 1.2815515594 * 2.0)) < 1e-6
    assert list(preds["horizon_step"]) == [1, 2]
