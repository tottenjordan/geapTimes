"""Validate the shipped config YAMLs against the schema."""

from pathlib import Path

from geaptimes.schemas import AutoMLParams, BQMLParams, ExperimentConfig, TimesFMParams

CONFIG_DIR = Path(__file__).parents[1] / "config"
CONFIG_PATH = CONFIG_DIR / "base_config.yaml"
COMPARISON_PATH = CONFIG_DIR / "comparison_config.yaml"


def test_base_config_loads_and_validates(monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "unit-test-proj")
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)

    assert cfg.project.id == "unit-test-proj"
    assert cfg.project.gcs_bucket == "geaptimes-unit-test-proj"
    assert cfg.forecast.horizon == 14
    assert cfg.data.station_filter.top_n == 25
    assert cfg.data.covariates.available_at_forecast_columns == [
        "temp",
        "prcp",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    by_name = {m.name: m for m in cfg.models}
    assert isinstance(by_name["timesfm"].params, TimesFMParams)
    assert by_name["timesfm"].params.use_covariates is False
    assert isinstance(by_name["bqml_arima_xreg"].params, BQMLParams)
    assert isinstance(by_name["automl"].params, AutoMLParams)
    assert by_name["automl"].enabled is False

    # Stage 4 pipeline block: cheap default keeps the endpoint transient.
    assert cfg.pipeline.serving.mode == "endpoint"
    assert cfg.pipeline.serving.keep_deployed is False


def test_comparison_config_enables_all_three_at_floor_budget(monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "unit-test-proj")
    cfg = ExperimentConfig.from_yaml(COMPARISON_PATH)

    # Same data shape as base_config -> same slug-named prepped/train/infer tables.
    assert cfg.data.station_filter.top_n == 25
    assert cfg.forecast.horizon == 14
    assert cfg.data.splits.test_length == 14
    assert cfg.execution.experiment_name == "citibike-daily-baseline"
    assert cfg.execution.target == "cloud_pipeline"

    by_name = {m.name: m for m in cfg.models}
    assert all(by_name[n].enabled for n in ("timesfm", "bqml_arima_xreg", "automl"))
    assert isinstance(by_name["automl"].params, AutoMLParams)
    assert by_name["automl"].params.budget_milli_node_hours == 1000
