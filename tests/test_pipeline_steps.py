"""Offline tests for the pipeline step functions (fakes for all cloud seams)."""

from datetime import datetime

import numpy as np
import pandas as pd

from geaptimes.experiment.tracking import ExperimentTracker
from geaptimes.models.base import PREDICTION_COLUMNS, Forecaster, ForecastResult
from geaptimes.pipelines import steps
from geaptimes.schemas import ExperimentConfig, ModelConfig

HORIZON = 3
CONTEXT = 32

BASE = {
    "project": {"id": "p", "gcs_bucket": "b", "bq_dataset": "ds", "region": "us-central1"},
    "forecast": {"horizon": HORIZON},
    "models": [
        {"name": "timesfm", "params": {"type": "timesfm", "context_len": CONTEXT}},
        {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
    ],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def _frame(n_series: int = 2, days: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=days, freq="D")
    frames = []
    for s in range(n_series):
        df = pd.DataFrame(
            {
                "start_station_name": f"s{s}",
                "date": [d.date() for d in dates],
                "num_trips": np.arange(days, dtype=float) + s,
                "splits": "TRAIN",
            }
        )
        df.loc[df.index[-HORIZON:], "splits"] = "TEST"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# -- build_tables ---------------------------------------------------------------------------------
def test_build_tables_step_runs_train_and_infer() -> None:
    seen: list[str] = []
    names = steps.build_tables_step(_cfg(), query_runner=seen.append)
    assert names["train"] == "p.ds.citibike_daily_train__top25_h3_t14_v14"
    assert names["infer"] == "p.ds.citibike_daily_infer__top25_h3_t14_v14"
    assert len(seen) == 2
    assert "CREATE OR REPLACE TABLE" in seen[0]
    assert "splits != 'TEST'" in seen[0]
    assert "splits = 'TEST'" in seen[1]


# -- series_contexts / responses_to_frame ---------------------------------------------------------
def test_series_contexts_excludes_test_and_truncates() -> None:
    cfg = _cfg()
    series_ids, contexts, last_dates = steps.series_contexts(_frame(n_series=2, days=80), cfg)
    assert series_ids == ["s0", "s1"]
    assert all(len(c) == CONTEXT for c in contexts)  # truncated to context_len
    # last non-TEST date = day (80 - HORIZON) -> 2018-01-01 + (80-3-1) days
    assert last_dates[0] == pd.Timestamp("2018-01-01") + pd.Timedelta(days=80 - HORIZON - 1)


def test_responses_to_frame_maps_by_series() -> None:
    last = pd.Timestamp("2018-03-01")
    responses = [
        {
            "series_id": "s0",
            "horizon": HORIZON,
            "forecast": [1.0, 2.0, 3.0],
            "q10": [0.5, 1.5, 2.5],
            "q30": [0.7, 1.7, 2.7],
            "q50": [1.0, 2.0, 3.0],
            "q70": [1.2, 2.2, 3.2],
            "q90": [1.5, 2.5, 3.5],
        }
    ]
    frame = steps.responses_to_frame(responses, {"s0": last}, model_name="timesfm")
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert len(frame) == HORIZON
    assert frame.iloc[0]["date"] == (last + pd.Timedelta(days=1)).date()
    assert frame.iloc[0]["forecast"] == 1.0
    assert frame.iloc[-1]["q90"] == 3.5


# -- run_backend ----------------------------------------------------------------------------------
class _FakeForecaster(Forecaster):
    def __init__(
        self, model_cfg: ModelConfig, cfg: ExperimentConfig, predictions: pd.DataFrame
    ) -> None:
        super().__init__(model_cfg, cfg)
        self._predictions = predictions
        self.fitted = False

    def fit(self) -> None:
        self.fitted = True

    def predict(self) -> ForecastResult:
        return ForecastResult(predictions=self._predictions, metadata={"backend": self.name})


class _TrackingAip:
    """Records the Vertex experiment lifecycle calls the tracker makes."""

    def __init__(self) -> None:
        self.params: list[dict] = []
        self.metrics: list[dict] = []
        self.calls: list[tuple] = []

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


def _predictions_frame() -> pd.DataFrame:
    rows = [
        {
            "series": "s0",
            "date": (pd.Timestamp("2018-03-18") + pd.Timedelta(days=step)).date(),
            "horizon_step": step + 1,
            "forecast": 10.0 + step,
            "q10": 9.0,
            "q30": 9.5,
            "q50": 10.0,
            "q70": 10.5,
            "q90": 11.0,
            "model": "bqml_arima_xreg",
        }
        for step in range(HORIZON)
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _actuals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_station_name": ["s0"] * HORIZON,
            "date": [
                (pd.Timestamp("2018-03-18") + pd.Timedelta(days=s)).date() for s in range(HORIZON)
            ],
            "num_trips": [10.0, 12.0, 12.0],
        }
    )


def test_run_backend_step_tracks_and_returns_result() -> None:
    cfg = _cfg()
    fake = _FakeForecaster(cfg.models[1], cfg, _predictions_frame())
    aip = _TrackingAip()
    writes: list[tuple[str, str]] = []
    tracker = ExperimentTracker(
        cfg, aiplatform=aip, artifact_sink=lambda uri, text: writes.append((uri, text))
    )
    out = steps.run_backend_step(
        cfg,
        "bqml_arima_xreg",
        forecaster=fake,
        actuals_loader=lambda _c: _actuals_frame(),
        tracker=tracker,
        now=lambda: datetime(2018, 3, 18, 12, 0, 0),  # noqa: DTZ001 - fixed clock for test
    )
    assert fake.fitted is True
    assert out.record.model == "bqml_arima_xreg"
    assert out.record.metrics["n_points"] == HORIZON
    assert out.record.artifact_uri.startswith("gs://b/experiments/")
    assert aip.metrics  # metrics logged
    assert aip.params[0]["backend"] == "bqml"  # from _run_params via model_cfg.params.type
    assert len(writes) == 2  # config.json + predictions.csv
    assert len(out.predictions) == HORIZON


# -- serving steps via a fake aiplatform ----------------------------------------------------------
class _FakeModel:
    def __init__(self, resource_name: str = "projects/x/models/m") -> None:
        self.resource_name = resource_name
        self.deployed: dict | None = None
        self.batch_kwargs: dict | None = None
        self.deleted = False

    def deploy(self, **kw: object) -> "_FakeEndpoint":
        self.deployed = kw
        return _FakeEndpoint()

    def batch_predict(self, **kw: object) -> str:
        self.batch_kwargs = kw
        return "BATCH_JOB"

    def delete(self) -> None:
        self.deleted = True


class _FakeEndpoint:
    def __init__(self, resource_name: str = "projects/x/endpoints/e") -> None:
        self.resource_name = resource_name
        self.undeployed = False
        self.deleted = False
        self.predict_instances: list | None = None

    def predict(self, *, instances: list) -> object:
        self.predict_instances = instances
        preds = [
            {
                "series_id": inst["series_id"],
                "horizon": HORIZON,
                "forecast": [1.0, 2.0, 3.0],
                "q10": [0.5, 1.5, 2.5],
                "q30": [0.7, 1.7, 2.7],
                "q50": [1.0, 2.0, 3.0],
                "q70": [1.2, 2.2, 3.2],
                "q90": [1.5, 2.5, 3.5],
            }
            for inst in instances
        ]
        return type("P", (), {"predictions": preds})()

    def undeploy_all(self) -> None:
        self.undeployed = True

    def delete(self) -> None:
        self.deleted = True


class _FakeAip:
    def __init__(self) -> None:
        self.init_kwargs: dict | None = None
        self.upload_kwargs: dict | None = None
        self.model = _FakeModel()
        self.endpoint = _FakeEndpoint()
        outer = self

        class Model:
            def __init__(self, resource_name: str) -> None:
                self.resource_name = resource_name

            def deploy(self, **kw: object) -> _FakeEndpoint:
                return outer.model.deploy(**kw)

            def batch_predict(self, **kw: object) -> str:
                return outer.model.batch_predict(**kw)

            def delete(self) -> None:
                outer.model.delete()

            @staticmethod
            def upload(**kw: object) -> _FakeModel:
                outer.upload_kwargs = kw
                return outer.model

        class Endpoint:
            def __init__(self, resource_name: str) -> None:
                outer.endpoint.resource_name = resource_name

            def predict(self, *, instances: list) -> object:
                return outer.endpoint.predict(instances=instances)

            def undeploy_all(self) -> None:
                outer.endpoint.undeploy_all()

            def delete(self) -> None:
                outer.endpoint.delete()

        self.Model = Model
        self.Endpoint = Endpoint

    def init(self, **kw: object) -> None:
        self.init_kwargs = kw


def test_register_timesfm_step_uploads_with_contract() -> None:
    aip = _FakeAip()
    name = steps.register_timesfm_step(_cfg(), aiplatform=aip)
    assert name == "projects/x/models/m"
    kw = aip.upload_kwargs
    assert kw is not None
    assert kw["serving_container_predict_route"] == "/predict"
    assert kw["serving_container_health_route"] == "/health"
    assert kw["serving_container_ports"] == [8080]
    assert kw["serving_container_image_uri"].endswith("geaptimes-runtime:latest")
    assert kw["serving_container_environment_variables"]["TIMESFM_HORIZON"] == str(HORIZON)
    assert kw["labels"] == {"solution": "geaptimes"}


def test_deploy_endpoint_step() -> None:
    aip = _FakeAip()
    name = steps.deploy_endpoint_step(_cfg(), "projects/x/models/m", aiplatform=aip)
    assert name == "projects/x/endpoints/e"
    deployed = aip.model.deployed
    assert deployed is not None
    assert deployed["machine_type"] == "n1-standard-4"
    assert deployed["traffic_percentage"] == 100


def test_endpoint_predict_step_builds_frame() -> None:
    aip = _FakeAip()
    frame = steps.endpoint_predict_step(
        _cfg(),
        "projects/x/endpoints/e",
        aiplatform=aip,
        frame_loader=lambda _c: _frame(n_series=2, days=80),
    )
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert len(frame) == 2 * HORIZON  # 2 series x horizon
    assert aip.endpoint.predict_instances is not None
    assert {row["series_id"] for row in aip.endpoint.predict_instances} == {"s0", "s1"}


def test_batch_predict_timesfm_step_builds_frame() -> None:
    aip = _FakeAip()

    def reader(_job: object) -> list[dict]:
        return [
            {
                "series_id": "s0",
                "horizon": HORIZON,
                "forecast": [1.0, 2.0, 3.0],
                "q10": [0.5, 1.5, 2.5],
                "q30": [0.7, 1.7, 2.7],
                "q50": [1.0, 2.0, 3.0],
                "q70": [1.2, 2.2, 3.2],
                "q90": [1.5, 2.5, 3.5],
            }
        ]

    frame = steps.batch_predict_timesfm_step(
        _cfg(),
        "projects/x/models/m",
        aiplatform=aip,
        frame_loader=lambda _c: _frame(n_series=1, days=80),
        instances_writer=lambda _c, _inst: "gs://b/instances.jsonl",
        predictions_reader=reader,
    )
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert len(frame) == HORIZON
    batch_kwargs = aip.model.batch_kwargs
    assert batch_kwargs is not None
    assert batch_kwargs["gcs_source"] == "gs://b/instances.jsonl"
    assert batch_kwargs["predictions_format"] == "jsonl"


def test_teardown_endpoint_step_deletes_endpoint_and_model() -> None:
    aip = _FakeAip()
    steps.teardown_endpoint_step(
        _cfg(), "projects/x/endpoints/e", model_resource_name="projects/x/models/m", aiplatform=aip
    )
    assert aip.endpoint.undeployed is True
    assert aip.endpoint.deleted is True
    assert aip.model.deleted is True


# -- compare --------------------------------------------------------------------------------------
def test_compare_step_ranks_by_rmse_then_mae() -> None:
    results = [
        ("timesfm", {"mae": 80.0, "rmse": 110.0}),
        ("bqml_arima_xreg", {"mae": 77.0, "rmse": 101.0}),
        ("automl", {"mae": 70.0, "rmse": 101.0}),
    ]
    comparison = steps.compare_step(results)
    # bqml and automl tie on rmse (101); automl wins on lower mae
    assert comparison.winner == "automl"
    assert [r["model"] for r in comparison.ranking] == ["automl", "bqml_arima_xreg", "timesfm"]
