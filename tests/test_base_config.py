"""Validate the shipped base_config.yaml against the schema."""

from pathlib import Path

from geaptimes.schemas import AutoMLParams, BQMLParams, ExperimentConfig, TimesFMParams

CONFIG_PATH = Path(__file__).parents[1] / "config" / "base_config.yaml"


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
