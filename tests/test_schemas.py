"""Tests for the typed experiment configuration schema."""

import pytest
from pydantic import ValidationError

from geaptimes.schemas import AutoMLParams, ExperimentConfig, TimesFMParams

VALID = {
    "project": {"id": "proj", "gcs_bucket": "bkt"},
    "models": [
        {"name": "timesfm", "params": {"type": "timesfm"}},
        {"name": "bqml", "params": {"type": "bqml"}},
    ],
    "execution": {"experiment_name": "exp"},
}


def test_valid_config_parses_with_defaults() -> None:
    cfg = ExperimentConfig.model_validate(VALID)
    assert cfg.forecast.horizon == 14
    assert cfg.data.station_filter.top_n == 25
    assert cfg.data.holiday_region == "US"
    # discriminated union resolves to the right concrete type
    assert isinstance(cfg.models[0].params, TimesFMParams)
    assert cfg.models[0].params.context_len == 512


def test_unknown_discriminator_rejected() -> None:
    bad = {**VALID, "models": [{"name": "x", "params": {"type": "prophet"}}]}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_context_len_must_be_multiple_of_32() -> None:
    bad = {
        **VALID,
        "models": [{"name": "t", "params": {"type": "timesfm", "context_len": 100}}],
    }
    with pytest.raises(ValidationError, match="multiple of 32"):
        ExperimentConfig.model_validate(bad)


def test_extra_keys_forbidden() -> None:
    bad = {**VALID, "surprise": True}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_non_positive_horizon_rejected() -> None:
    bad = {**VALID, "forecast": {"horizon": 0}}
    with pytest.raises(ValidationError, match="horizon"):
        ExperimentConfig.model_validate(bad)


def test_from_yaml_with_env_substitution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "my-proj")
    yaml_text = (
        "project:\n"
        "  id: ${GCP_PROJECT}\n"
        "  gcs_bucket: geaptimes-${GCP_PROJECT}\n"
        "models:\n"
        "  - name: timesfm\n"
        "    params: {type: timesfm}\n"
        "execution:\n"
        "  experiment_name: exp\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.project.id == "my-proj"
    assert cfg.project.gcs_bucket == "geaptimes-my-proj"


def test_from_yaml_missing_env_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    path = tmp_path / "cfg.yaml"
    path.write_text("project:\n  id: ${DOES_NOT_EXIST}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not set"):
        ExperimentConfig.from_yaml(path)


def test_doe_defaults_to_empty_axes() -> None:
    cfg = ExperimentConfig.model_validate(VALID)
    assert cfg.doe.axes == {}


def test_doe_axes_round_trip() -> None:
    cfg = ExperimentConfig.model_validate({**VALID, "doe": {"axes": {"forecast.horizon": [7, 14]}}})
    assert cfg.doe.axes == {"forecast.horizon": [7, 14]}


def test_automl_budget_defaults_and_validates() -> None:
    cfg = ExperimentConfig.model_validate(
        {**VALID, "models": [{"name": "a", "params": {"type": "automl"}}]}
    )
    params = cfg.models[0].params
    assert isinstance(params, AutoMLParams)
    assert params.budget_milli_node_hours == 1000


def test_automl_budget_must_be_positive() -> None:
    bad = {
        **VALID,
        "models": [{"name": "a", "params": {"type": "automl", "budget_milli_node_hours": 0}}],
    }
    with pytest.raises(ValidationError, match="budget_milli_node_hours"):
        ExperimentConfig.model_validate(bad)
