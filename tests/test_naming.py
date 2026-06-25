"""Tests for deterministic table naming."""

from geaptimes.naming import config_fingerprint, config_slug, data_slug, table_names
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "p", "gcs_bucket": "b"},
    "models": [{"name": "t", "params": {"type": "timesfm"}}],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def test_slugs_default() -> None:
    cfg = _cfg()
    assert data_slug(cfg.data) == "top25"
    assert config_slug(cfg.data, cfg.forecast) == "top25_h14_t14_v14"


def test_table_names_default() -> None:
    names = table_names(_cfg())
    assert names["source"] == "citibike_daily_source__top25"
    assert names["prepped"] == "citibike_daily_prepped__top25_h14_t14_v14"
    assert names["train"] == "citibike_daily_train__top25_h14_t14_v14"
    assert names["infer"] == "citibike_daily_infer__top25_h14_t14_v14"


def test_config_slug_tracks_splits_and_horizon() -> None:
    cfg = _cfg(forecast={"horizon": 7}, data={"splits": {"test_length": 30, "validate_length": 7}})
    assert config_slug(cfg.data, cfg.forecast) == "top25_h7_t30_v7"


def test_fingerprint_is_stable_and_includes_experiment() -> None:
    fp1 = config_fingerprint(_cfg())
    fp2 = config_fingerprint(_cfg())
    assert fp1 == fp2
    assert '"experiment":"exp"' in fp1
