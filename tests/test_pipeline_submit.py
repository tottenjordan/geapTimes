"""Offline tests for the pipeline submit path (fake aiplatform; real compile, no cloud)."""

import logging
from pathlib import Path

import yaml

from geaptimes.pipelines import submit
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "proj", "gcs_bucket": "bkt", "bq_dataset": "ds", "region": "us-central1"},
    "execution": {"experiment_name": "citibike-daily-baseline"},
    "models": [
        {"name": "timesfm", "params": {"type": "timesfm"}},
        {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
        {"name": "automl", "enabled": False, "params": {"type": "automl"}},
    ],
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


class _FakeJob:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.submitted_experiment: str | None = None

    def submit(self, *, experiment: str) -> None:
        self.submitted_experiment = experiment


class _FakeAip:
    def __init__(self) -> None:
        self.init_kwargs: dict | None = None
        self.job: _FakeJob | None = None
        outer = self

        def pipeline_job(**kwargs: object) -> _FakeJob:
            outer.job = _FakeJob(**kwargs)
            return outer.job

        self.PipelineJob = pipeline_job

    def init(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs


# -- with_automl_enabled --------------------------------------------------------------------------
def test_with_automl_enabled_flips_existing_model() -> None:
    out = submit.with_automl_enabled(_cfg())
    automl = next(m for m in out.models if m.params.type == "automl")
    assert automl.enabled is True


def test_with_automl_enabled_does_not_mutate_input() -> None:
    base = _cfg()
    submit.with_automl_enabled(base)
    original_automl = next(m for m in base.models if m.params.type == "automl")
    assert original_automl.enabled is False  # input untouched


def test_with_automl_enabled_appends_when_absent() -> None:
    cfg = _cfg(
        models=[
            {"name": "timesfm", "params": {"type": "timesfm"}},
            {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
        ]
    )
    out = submit.with_automl_enabled(cfg)
    automl = [m for m in out.models if m.params.type == "automl"]
    assert len(automl) == 1
    assert automl[0].enabled is True


# -- submit_pipeline ------------------------------------------------------------------------------
def test_submit_pipeline_compiles_and_submits(tmp_path: Path) -> None:
    aip = _FakeAip()
    template = tmp_path / "pipeline.yaml"
    job = submit.submit_pipeline(_cfg(), aiplatform=aip, template_path=template)

    # compiled the DAG to the given path (real compile, no cloud)
    assert template.exists()
    # initialized aiplatform with project/location/staging bucket
    assert aip.init_kwargs == {
        "project": "proj",
        "location": "us-central1",
        "staging_bucket": "gs://bkt",
    }
    # PipelineJob constructed with the expected contract
    assert job.kwargs["display_name"] == "geaptimes-comparison__top25_h14_t14_v14"
    assert job.kwargs["template_path"] == str(template)
    assert job.kwargs["pipeline_root"] == "gs://bkt/pipeline_root/citibike-daily-baseline"
    assert job.kwargs["labels"] == {"solution": "geaptimes"}
    assert "config_json" in job.kwargs["parameter_values"]
    # submitted under the experiment
    assert job.submitted_experiment == "citibike-daily-baseline"


def test_submit_pipeline_caching_override(tmp_path: Path) -> None:
    aip = _FakeAip()
    job = submit.submit_pipeline(
        _cfg(), aiplatform=aip, template_path=tmp_path / "p.yaml", enable_caching=False
    )
    assert job.kwargs["enable_caching"] is False


# -- CLI ------------------------------------------------------------------------------------------
def test_cli_dry_run_compiles_without_submit(tmp_path: Path, caplog) -> None:
    out = tmp_path / "compiled.yaml"
    with caplog.at_level(logging.INFO):
        rc = submit.main(["--config", _write_config(tmp_path), "--dry-run", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert "not submitted" in " ".join(caplog.messages)


def _write_config(tmp_path: Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    return str(path)
