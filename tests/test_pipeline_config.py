"""Tests for the pure pipeline config derivations."""

from geaptimes.pipelines import config as pcfg
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "proj", "gcs_bucket": "bkt", "region": "us-central1", "bq_dataset": "ds"},
    "models": [{"name": "timesfm", "params": {"type": "timesfm"}}],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def test_image_uri() -> None:
    assert (
        pcfg.image_uri(_cfg())
        == "us-central1-docker.pkg.dev/proj/geaptimes/geaptimes-runtime:latest"
    )


def test_pipeline_root_default() -> None:
    assert pcfg.pipeline_root(_cfg()) == "gs://bkt/pipeline_root/exp"


def test_display_names_use_config_slug() -> None:
    cfg = _cfg()
    assert pcfg.timesfm_model_display_name(cfg) == "timesfm__top25_h14_t14_v14"
    assert pcfg.timesfm_endpoint_display_name(cfg) == "timesfm-endpoint__top25_h14_t14_v14"
    assert pcfg.pipeline_job_display_name(cfg) == "geaptimes-comparison__top25_h14_t14_v14"


def test_resource_labels() -> None:
    assert pcfg.resource_labels() == {"solution": "geaptimes"}
