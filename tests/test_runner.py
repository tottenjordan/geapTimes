"""Offline tests for run_experiment (fake forecasters/tracker/actuals; no cloud, no models)."""

from datetime import UTC, datetime

import pandas as pd

from geaptimes.experiment.runner import ForecasterBuilder, _safe_run_name, run_experiment
from geaptimes.experiment.tracking import ExperimentTracker
from geaptimes.models.base import (
    DATE_COLUMN,
    FORECAST_COLUMN,
    MODEL_COLUMN,
    PREDICTION_COLUMNS,
    QUANTILE_COLUMNS,
    SERIES_COLUMN,
    Forecaster,
    ForecastResult,
)
from geaptimes.schemas import ExperimentConfig, ModelConfig

BASE = {
    "project": {"id": "p", "gcs_bucket": "b", "region": "us-central1"},
    "models": [{"name": "timesfm", "params": {"type": "timesfm"}}],
    "execution": {"experiment_name": "exp"},
}

FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
DATES = ["2018-05-18", "2018-05-19"]


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def _predictions() -> pd.DataFrame:
    rows = []
    for step, date in enumerate(DATES, start=1):
        row = {
            SERIES_COLUMN: "s",
            DATE_COLUMN: pd.Timestamp(date).date(),
            "horizon_step": step,
            FORECAST_COLUMN: 10.0,
            MODEL_COLUMN: "m",
        }
        row.update(dict.fromkeys(QUANTILE_COLUMNS, 10.0))
        rows.append(row)
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _actuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_station_name": ["s", "s"],
            "date": pd.to_datetime(DATES),
            "num_trips": [12.0, 10.0],
        }
    )


class FakeForecaster(Forecaster):
    """Canned forecaster: records fit() and returns a fixed ForecastResult."""

    def __init__(self, model_cfg: ModelConfig, cfg: ExperimentConfig) -> None:
        super().__init__(model_cfg, cfg)
        self.fitted = False

    def fit(self) -> None:
        self.fitted = True

    def predict(self) -> ForecastResult:
        return ForecastResult(predictions=_predictions(), metadata={"device": "cpu"})


class FakeAip:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.params: list[dict] = []
        self.metrics: list[dict] = []

    def init(self, **kw: object) -> None:
        self.calls.append(("init", kw))

    def start_run(self, name: str) -> None:
        self.calls.append(("start_run", name))

    def end_run(self) -> None:
        self.calls.append(("end_run",))

    def log_params(self, params: dict) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.append(metrics)


def _harness(
    cfg: ExperimentConfig,
) -> tuple[
    ExperimentTracker, FakeAip, list[tuple[str, str]], list[FakeForecaster], ForecasterBuilder
]:
    aip = FakeAip()
    writes: list[tuple[str, str]] = []
    built: list[FakeForecaster] = []
    tracker = ExperimentTracker(
        cfg, aiplatform=aip, artifact_sink=lambda uri, text: writes.append((uri, text))
    )

    def build(config: ExperimentConfig) -> list[Forecaster]:
        fc = FakeForecaster(config.models[0], config)
        built.append(fc)
        return [fc]

    return tracker, aip, writes, built, build


def test_single_point_single_model_run() -> None:
    cfg = _cfg()
    tracker, aip, writes, built, build = _harness(cfg)
    records = run_experiment(
        cfg,
        tracker=tracker,
        forecasters=build,
        actuals_loader=lambda _c: _actuals(),
        now=lambda: FIXED_NOW,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec.run_name == "timesfm-base-20260102-030405"
    assert rec.model == "timesfm"
    assert rec.metrics["mae"] == 1.0  # errors -2, 0
    # forecaster metadata is preserved; the per-series breakdown from the eval suite is attached.
    assert rec.metadata["device"] == "cpu"
    assert [row["series"] for row in rec.metadata["per_series"]] == ["s"]
    assert rec.artifact_uri == "gs://b/experiments/exp/timesfm-base-20260102-030405"
    assert built[0].fitted  # fit() was called
    # tracker lifecycle + logging happened
    assert ("start_run", "timesfm-base-20260102-030405") in aip.calls
    assert aip.calls[-1] == ("end_run",)
    assert aip.params[0]["backend"] == "timesfm"
    assert aip.params[0]["horizon"] == 14
    assert len(writes) == 2  # config.json + predictions.csv


def test_doe_grid_expands_runs() -> None:
    cfg = _cfg(doe={"axes": {"forecast.horizon": [7, 14]}})
    tracker, _aip, _writes, _built, build = _harness(cfg)
    records = run_experiment(
        cfg,
        tracker=tracker,
        forecasters=build,
        actuals_loader=lambda _c: _actuals(),
        now=lambda: FIXED_NOW,
    )
    assert len(records) == 2
    assert {r.overrides["forecast.horizon"] for r in records} == {7, 14}


def test_safe_run_name_sanitizes() -> None:
    assert _safe_run_name("bqml_arima_xreg-base-20260102") == "bqml-arima-xreg-base-20260102"
