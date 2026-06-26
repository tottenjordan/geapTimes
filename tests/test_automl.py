"""Offline tests for AutoMLForecaster (config translation + fit wiring; no Vertex calls)."""

import pandas as pd
import pytest

from geaptimes.models.automl import AutoMLForecaster
from geaptimes.models.base import PREDICTION_COLUMNS, QUANTILES
from geaptimes.schemas import ExperimentConfig

ATTR = ["capacity", "region_id", "latitude", "longitude"]
AVAIL = ["temp", "prcp", "day_of_week", "month", "is_weekend"]
UNAVAIL = ["avg_tripduration", "pct_subscriber", "ratio_gender"]

BASE = {
    "project": {"id": "p", "gcs_bucket": "b", "bq_dataset": "ds", "region": "us-central1"},
    "data": {
        "covariates": {
            "time_series_attribute_columns": ATTR,
            "available_at_forecast_columns": AVAIL,
            "unavailable_at_forecast_columns": UNAVAIL,
        }
    },
    "models": [{"name": "automl", "params": {"type": "automl", "context_window": 28}}],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def _forecaster(**over: object) -> AutoMLForecaster:
    cfg = _cfg(**over)
    return AutoMLForecaster(cfg.models[0], cfg)


class FakeAip:
    """Records the Vertex SDK calls the forecaster makes."""

    def __init__(self) -> None:
        self.init_kwargs: dict | None = None
        self.ds_kwargs: dict | None = None
        self.job_kwargs: dict | None = None
        self.run_kwargs: dict | None = None
        outer = self

        class TimeSeriesDataset:
            @staticmethod
            def create(**kw: object) -> str:
                outer.ds_kwargs = kw
                return "DATASET"

        class AutoMLForecastingTrainingJob:
            def __init__(self, **kw: object) -> None:
                outer.job_kwargs = kw

            def run(self, **kw: object) -> str:
                outer.run_kwargs = kw
                return "MODEL"

        self.TimeSeriesDataset = TimeSeriesDataset
        self.AutoMLForecastingTrainingJob = AutoMLForecastingTrainingJob

    def init(self, **kw: object) -> None:
        self.init_kwargs = kw


def test_training_job_kwargs() -> None:
    kw = _forecaster().training_job_kwargs()
    assert kw["display_name"] == "automl__top25_h14_t14_v14"
    assert kw["optimization_objective"] == "minimize-rmse"
    assert kw["labels"] == {"solution": "geaptimes"}


def test_training_job_kwargs_column_specs_cover_target_and_roles() -> None:
    specs = _forecaster().training_job_kwargs()["column_specs"]
    # Every used column gets an explicit "auto" transformation -- crucially the target, which the
    # SDK's default auto-generation omits (Vertex Forecasting then rejects the job).
    assert specs["num_trips"] == "auto"
    for col in ("date", "start_station_name", *ATTR, *AVAIL, *UNAVAIL):
        assert specs[col] == "auto"
    assert set(specs) == {"num_trips", "date", "start_station_name", *ATTR, *AVAIL, *UNAVAIL}


def test_run_kwargs_maps_column_roles_and_settings() -> None:
    kw = _forecaster().run_kwargs()
    assert kw["target_column"] == "num_trips"
    assert kw["time_column"] == "date"
    assert kw["time_series_identifier_column"] == "start_station_name"
    assert kw["time_series_attribute_columns"] == ATTR
    assert kw["available_at_forecast_columns"] == AVAIL
    # Vertex requires the target column itself in the unavailable-at-forecast set.
    assert kw["unavailable_at_forecast_columns"] == [*UNAVAIL, "num_trips"]
    assert kw["forecast_horizon"] == 14
    assert kw["context_window"] == 28
    assert kw["budget_milli_node_hours"] == 1000
    assert kw["data_granularity_unit"] == "day"
    assert kw["data_granularity_count"] == 1
    # No predefined split: Vertex auto-splits the TEST-free train table chronologically.
    assert "predefined_split_column_name" not in kw
    assert kw["quantiles"] == QUANTILES
    assert kw["holiday_regions"] == ["US"]
    assert kw["model_labels"] == {"solution": "geaptimes"}


def test_holiday_regions_none_when_unset() -> None:
    kw = _forecaster(
        data={"holiday_region": None, "covariates": {"available_at_forecast_columns": AVAIL}}
    ).run_kwargs()
    assert kw["holiday_regions"] is None


def test_batch_predict_kwargs() -> None:
    kw = _forecaster().batch_predict_kwargs()
    assert kw["bigquery_source"] == "bq://p.ds.citibike_daily_infer__top25_h14_t14_v14"
    assert kw["bigquery_destination_prefix"] == "bq://p.ds"
    assert kw["instances_format"] == "bigquery"
    assert kw["predictions_format"] == "bigquery"
    assert kw["labels"] == {"solution": "geaptimes"}


def test_fit_wires_dataset_job_and_run() -> None:
    cfg = _cfg()
    fake = FakeAip()
    fc = AutoMLForecaster(cfg.models[0], cfg, aiplatform=fake)
    fc.fit()

    assert fake.init_kwargs == {"project": "p", "location": "us-central1"}
    assert fake.ds_kwargs is not None
    assert fake.job_kwargs is not None
    assert fake.run_kwargs is not None
    assert fake.ds_kwargs["bq_source"] == "bq://p.ds.citibike_daily_train__top25_h14_t14_v14"
    assert fake.ds_kwargs["labels"] == {"solution": "geaptimes"}
    assert fake.job_kwargs["display_name"] == "automl__top25_h14_t14_v14"
    assert fake.run_kwargs["dataset"] == "DATASET"
    assert fake.run_kwargs["forecast_horizon"] == 14
    assert fc.model == "MODEL"


def test_predict_maps_flattened_output() -> None:
    cfg = _cfg()
    raw = pd.DataFrame(
        {
            "start_station_name": ["s0", "s0"],
            "date": pd.to_datetime(["2018-05-18", "2018-05-19"]),
            "value": [10.0, 11.0],
            "q10": [8.0, 9.0],
            "q30": [9.0, 10.0],
            "q50": [10.0, 11.0],
            "q70": [11.0, 12.0],
            "q90": [12.0, 13.0],
        }
    )
    fc = AutoMLForecaster(cfg.models[0], cfg, prediction_reader=lambda: raw)
    result = fc.predict()
    preds = result.predictions
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert result.metadata["objective"] == "minimize-rmse"
    first = preds.iloc[0]
    assert first["forecast"] == 10.0
    assert first["q90"] == 12.0
    assert list(preds["horizon_step"]) == [1, 2]


def test_predict_flattens_nested_vertex_struct() -> None:
    cfg = _cfg()
    # Vertex forecasting batch output: a nested predicted_<target> STRUCT per row. BigQuery yields
    # STRUCTs as dicts and ARRAYs as lists in pandas. Levels are intentionally out of order to
    # exercise the level->index matching.
    raw = pd.DataFrame(
        {
            "start_station_name": ["s0", "s0"],
            "date": pd.to_datetime(["2018-05-18", "2018-05-19"]),
            "predicted_num_trips": [
                {
                    "value": 10.0,
                    "quantile_values": [0.5, 0.1, 0.9, 0.3, 0.7],
                    "quantile_predictions": [10.0, 8.0, 12.0, 9.0, 11.0],
                },
                {
                    "value": 11.0,
                    "quantile_values": [0.5, 0.1, 0.9, 0.3, 0.7],
                    "quantile_predictions": [11.0, 9.0, 13.0, 10.0, 12.0],
                },
            ],
        }
    )
    fc = AutoMLForecaster(cfg.models[0], cfg, prediction_reader=lambda: raw)
    preds = fc.predict().predictions
    assert list(preds.columns) == PREDICTION_COLUMNS
    first = preds.iloc[0]
    assert first["forecast"] == 10.0
    # matched by level, not position: q10=8, q50=10, q90=12
    assert (first["q10"], first["q50"], first["q90"]) == (8.0, 10.0, 12.0)
    assert list(preds["horizon_step"]) == [1, 2]


def test_flatten_raises_on_missing_quantile_level() -> None:
    cfg = _cfg()
    raw = pd.DataFrame(
        {
            "start_station_name": ["s0"],
            "date": pd.to_datetime(["2018-05-18"]),
            "predicted_num_trips": [
                {
                    "value": 10.0,
                    "quantile_values": [0.1, 0.3, 0.5, 0.7],  # missing 0.9
                    "quantile_predictions": [8.0, 9.0, 10.0, 11.0],
                }
            ],
        }
    )
    fc = AutoMLForecaster(cfg.models[0], cfg, prediction_reader=lambda: raw)
    with pytest.raises(ValueError, match=r"quantile level 0\.9 not found"):
        fc.predict()
