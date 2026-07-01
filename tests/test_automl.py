"""Offline tests for AutoMLForecaster (config translation + fit wiring; no Vertex calls)."""

import math

import pandas as pd
import pytest

from geaptimes.constants import RESOURCE_LABELS
from geaptimes.models.automl import AutoMLForecaster, _label_bq_table
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
    assert kw["optimization_objective"] == "minimize-quantile-loss"
    assert kw["labels"] == {"solution": "geaptimes"}


def test_training_job_kwargs_column_specs_cover_target_and_roles() -> None:
    specs = _forecaster().training_job_kwargs()["column_specs"]
    # Every feature/target column gets an explicit "auto" transformation -- crucially the target,
    # which the SDK's default auto-generation omits (Vertex Forecasting then rejects the job).
    assert specs["num_trips"] == "auto"
    for col in ("date", *ATTR, *AVAIL, *UNAVAIL):
        assert specs[col] == "auto"
    # The series identifier must NOT carry a transformation (Vertex FailedPrecondition otherwise).
    assert "start_station_name" not in specs
    assert set(specs) == {"num_trips", "date", *ATTR, *AVAIL, *UNAVAIL}


def test_run_kwargs_maps_column_roles_and_settings() -> None:
    kw = _forecaster().run_kwargs()
    assert kw["target_column"] == "num_trips"
    assert kw["time_column"] == "date"
    assert kw["time_series_identifier_column"] == "start_station_name"
    assert kw["time_series_attribute_columns"] == ATTR
    # Vertex requires the time column itself in the available-at-forecast set.
    assert kw["available_at_forecast_columns"] == [*AVAIL, "date"]
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


def test_validate_config_passes_on_default() -> None:
    _forecaster().validate_config()  # no raise


def test_validate_config_collects_role_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    fc = _forecaster()
    broken = fc.run_kwargs()
    broken["available_at_forecast_columns"] = list(AVAIL)  # drop the appended time column
    broken["unavailable_at_forecast_columns"] = list(UNAVAIL)  # drop the appended target
    monkeypatch.setattr(fc, "run_kwargs", lambda: broken)
    with pytest.raises(ValueError) as exc:  # noqa: PT011 - message asserted below
        fc.validate_config()
    msg = str(exc.value)
    assert "time column 'date' must be in available_at_forecast_columns" in msg
    assert "target 'num_trips' must be in unavailable_at_forecast_columns" in msg


def test_validate_config_flags_untransformed_role_column(monkeypatch: pytest.MonkeyPatch) -> None:
    fc = _forecaster()
    monkeypatch.setattr(fc, "_column_specs", lambda: {"num_trips": "auto"})  # covariates dropped
    with pytest.raises(ValueError, match="missing a column_specs transformation"):
        fc.validate_config()


def _minimize_rmse_forecaster() -> AutoMLForecaster:
    return _forecaster(
        models=[
            {
                "name": "automl",
                "params": {"type": "automl", "optimization_objective": "minimize-rmse"},
            }
        ]
    )


def test_minimize_rmse_is_point_only_and_valid() -> None:
    # minimize-rmse targets the mean; Vertex allows quantiles only with minimize-quantile-loss, so
    # this run must request NO quantiles -- and that is a valid config (no raise).
    fc = _minimize_rmse_forecaster()
    assert "quantiles" not in fc.run_kwargs()
    assert fc.training_job_kwargs()["optimization_objective"] == "minimize-rmse"
    fc.validate_config()  # point-only is valid


def test_quantile_loss_objective_requests_quantiles() -> None:
    # the default objective still emits the standardized quantile levels.
    kw = _forecaster().run_kwargs()
    assert kw["quantiles"] == QUANTILES


def test_validate_config_flags_transformed_series_column(monkeypatch: pytest.MonkeyPatch) -> None:
    fc = _forecaster()
    specs = {**fc._column_specs(), "start_station_name": "auto"}  # noqa: SLF001
    monkeypatch.setattr(fc, "_column_specs", lambda: specs)
    with pytest.raises(ValueError, match="series column 'start_station_name' must not"):
        fc.validate_config()


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
    assert fake.run_kwargs["sync"] is True  # default; preflight flips to False
    assert fc.model == "MODEL"
    assert fc.dataset == "DATASET"  # handle captured for preflight cleanup
    assert fc.training_job is not None


class _FakeModel:
    resource_name = "projects/p/locations/us-central1/models/123"


class FakeAipModel(FakeAip):
    """Like :class:`FakeAip` but ``run`` returns a Model object exposing ``resource_name``."""

    def __init__(self) -> None:
        super().__init__()
        outer = self

        class AutoMLForecastingTrainingJob:
            def __init__(self, **kw: object) -> None:
                outer.job_kwargs = kw

            def run(self, **kw: object) -> _FakeModel:
                outer.run_kwargs = kw
                return _FakeModel()

        self.AutoMLForecastingTrainingJob = AutoMLForecastingTrainingJob


def test_model_reference_empty_before_fit_and_resource_name_after() -> None:
    cfg = _cfg()
    fc = AutoMLForecaster(cfg.models[0], cfg, aiplatform=FakeAipModel())
    assert fc.model_reference == ""  # nothing trained yet
    fc.fit()
    assert fc.model_reference == "projects/p/locations/us-central1/models/123"


def test_attach_model_resolves_model_by_reference() -> None:
    cfg = _cfg()
    resolved: dict[str, object] = {}

    class FakeAipAttach:
        def init(self, **kw: object) -> None:
            resolved["init"] = kw

        def Model(self, reference: str) -> str:  # noqa: N802 - mirrors aiplatform.Model
            resolved["reference"] = reference
            return f"MODEL[{reference}]"

    fc = AutoMLForecaster(cfg.models[0], cfg, aiplatform=FakeAipAttach())
    fc.attach_model("projects/p/locations/us-central1/models/123")
    assert resolved["init"] == {"project": "p", "location": "us-central1"}
    assert resolved["reference"] == "projects/p/locations/us-central1/models/123"
    # predict() now runs against the attached model without a re-fit (cheap infer retry).
    assert fc.model == "MODEL[projects/p/locations/us-central1/models/123]"


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
    assert result.metadata["objective"] == "minimize-quantile-loss"
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


def test_flatten_point_only_output_fills_nan_quantiles() -> None:
    # A minimize-rmse run emits no quantile fields (point-only); qXX must become NaN, not error,
    # so the standardized schema holds and the scorer reports quantile_loss as NaN.
    cfg = _cfg()
    raw = pd.DataFrame(
        {
            "start_station_name": ["s0", "s0"],
            "date": pd.to_datetime(["2018-05-18", "2018-05-19"]),
            "predicted_num_trips": [
                {"value": 10.0, "quantile_values": [], "quantile_predictions": []},
                {"value": 11.0},  # quantile fields entirely absent
            ],
        }
    )
    fc = AutoMLForecaster(cfg.models[0], cfg, prediction_reader=lambda: raw)
    preds = fc.predict().predictions
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert list(preds["forecast"]) == [10.0, 11.0]
    for level in ("q10", "q30", "q50", "q70", "q90"):
        assert all(math.isnan(v) for v in preds[level])


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


# -- output table resolution + labelling (live-run regressions, 4.9) -------------------------------
class _FakeOutputInfo:
    def __init__(self, dataset: str, table: str) -> None:
        self.bigquery_output_dataset = dataset
        self.bigquery_output_table = table


class _FakeBatchJob:
    def __init__(self, dataset: str, table: str) -> None:
        self.output_info = _FakeOutputInfo(dataset, table)


def test_output_table_id_uses_job_suffix() -> None:
    # Regression: Vertex names the table predictions_<ts>; we must read the suffix from the job,
    # never hardcode ".predictions" (which 404'd on the first live run).
    job = _FakeBatchJob("bq://p.ds", "predictions_2026_06_26T01_44_29_529Z_164")
    assert (
        AutoMLForecaster._output_table_id(job)  # noqa: SLF001 - exercising the resolution rule
        == "p.ds.predictions_2026_06_26T01_44_29_529Z_164"
    )


class _FakeBqTable:
    def __init__(self) -> None:
        self.labels: dict[str, str] = {}


class _FakeBqClient:
    def __init__(self) -> None:
        self.table = _FakeBqTable()
        self.updated_fields: list[str] | None = None

    def get_table(self, _table_id: str) -> _FakeBqTable:
        return self.table

    def update_table(self, _table: _FakeBqTable, fields: list[str]) -> None:
        self.updated_fields = fields


def test_label_bq_table_stamps_solution_label() -> None:
    client = _FakeBqClient()
    _label_bq_table(client, "p.ds.predictions_x")
    assert client.table.labels["solution"] == RESOURCE_LABELS["solution"]
    assert client.updated_fields == ["labels"]


def test_label_bq_table_is_best_effort() -> None:
    class _Boom:
        def get_table(self, _t: str) -> object:
            msg = "transient bq error"
            raise RuntimeError(msg)

    # A labelling failure must not propagate (governance nit, not a result error).
    _label_bq_table(_Boom(), "p.ds.predictions_x")
