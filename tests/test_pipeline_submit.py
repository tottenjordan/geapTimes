"""Offline tests for the pipeline submit path (fake aiplatform; real compile, no cloud)."""

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from geaptimes.pipelines import submit
from geaptimes.pipelines.config import serving_fingerprint
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


# -- with_automl_disabled -------------------------------------------------------------------------
def test_with_automl_disabled_flips_existing_model() -> None:
    enabled = _cfg(
        models=[
            {"name": "timesfm", "params": {"type": "timesfm"}},
            {"name": "automl", "enabled": True, "params": {"type": "automl"}},
        ]
    )
    out = submit.with_automl_disabled(enabled)
    automl = next(m for m in out.models if m.params.type == "automl")
    assert automl.enabled is False


def test_with_automl_disabled_does_not_mutate_input() -> None:
    enabled = _cfg(
        models=[
            {"name": "timesfm", "params": {"type": "timesfm"}},
            {"name": "automl", "enabled": True, "params": {"type": "automl"}},
        ]
    )
    submit.with_automl_disabled(enabled)
    original_automl = next(m for m in enabled.models if m.params.type == "automl")
    assert original_automl.enabled is True  # input untouched


def test_with_automl_disabled_noop_when_absent() -> None:
    cfg = _cfg(
        models=[
            {"name": "timesfm", "params": {"type": "timesfm"}},
            {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
        ]
    )
    out = submit.with_automl_disabled(cfg)
    assert [m.params.type for m in out.models] == ["timesfm", "bqml"]


# -- with_data_rebuild_forced ---------------------------------------------------------------------
def test_with_data_rebuild_forced_sets_flag() -> None:
    out = submit.with_data_rebuild_forced(_cfg())
    assert out.data.force_rebuild is True


def test_with_data_rebuild_forced_does_not_mutate_input() -> None:
    base = _cfg()
    submit.with_data_rebuild_forced(base)
    assert base.data.force_rebuild is False  # input untouched


# -- with_caching_disabled ------------------------------------------------------------------------
def test_with_caching_disabled_sets_flag() -> None:
    out = submit.with_caching_disabled(_cfg())
    assert out.pipeline.enable_caching is False


def test_with_caching_disabled_does_not_mutate_input() -> None:
    base = _cfg()
    submit.with_caching_disabled(base)
    assert base.pipeline.enable_caching is True  # input untouched (default on)


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


def _capture_compile(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    """Replace submit.compile_pipeline with a stub that records the reuse passthrough kwargs."""

    def fake_compile(
        _cfg: ExperimentConfig,
        out_path: object,
        *,
        reused_endpoint: str = "",
        serving_fingerprint: str = "",
    ) -> str:
        captured["reused_endpoint"] = reused_endpoint
        captured["serving_fingerprint"] = serving_fingerprint
        return str(out_path)

    monkeypatch.setattr(submit, "compile_pipeline", fake_compile)


def test_submit_pipeline_reuse_passes_resolved_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    _capture_compile(monkeypatch, captured)
    monkeypatch.setattr(submit, "resolve_image_digest_step", lambda _cfg: "sha256:abc")
    monkeypatch.setattr(
        submit,
        "find_reusable_endpoint_step",
        lambda _cfg, _fp, *, aiplatform: "projects/x/endpoints/warm",  # noqa: ARG005 - fake
    )
    cfg = _cfg(pipeline={"serving": {"reuse_endpoint": True}})
    submit.submit_pipeline(cfg, aiplatform=_FakeAip(), template_path=tmp_path / "p.yaml")
    assert captured["reused_endpoint"] == "projects/x/endpoints/warm"
    # the fingerprint is computed from the resolved digest and threaded through for label stamping
    assert captured["serving_fingerprint"] == serving_fingerprint(cfg, "sha256:abc")


def test_submit_pipeline_reuse_no_match_deploys_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    _capture_compile(monkeypatch, captured)
    monkeypatch.setattr(submit, "resolve_image_digest_step", lambda _cfg: "sha256:abc")
    monkeypatch.setattr(
        submit,
        "find_reusable_endpoint_step",
        lambda _cfg, _fp, *, aiplatform: "",  # noqa: ARG005 - fake: no warm endpoint
    )
    cfg = _cfg(pipeline={"serving": {"reuse_endpoint": True}})
    submit.submit_pipeline(cfg, aiplatform=_FakeAip(), template_path=tmp_path / "p.yaml")
    assert captured["reused_endpoint"] == ""  # falls through to the fresh-deploy path
    assert captured["serving_fingerprint"]  # still stamped so the fresh deploy is discoverable


def test_submit_pipeline_no_reuse_makes_no_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    _capture_compile(monkeypatch, captured)

    def _boom(*_a: object, **_k: object) -> str:
        msg = "reuse lookup must not run when reuse_endpoint is off"
        raise AssertionError(msg)

    monkeypatch.setattr(submit, "resolve_image_digest_step", _boom)
    monkeypatch.setattr(submit, "find_reusable_endpoint_step", _boom)
    submit.submit_pipeline(_cfg(), aiplatform=_FakeAip(), template_path=tmp_path / "p.yaml")
    assert captured["reused_endpoint"] == ""
    assert captured["serving_fingerprint"] == ""


def test_submit_pipeline_pipeline_level_caching_always_off(tmp_path: Path) -> None:
    # The Vertex pipeline-level enable_caching is always submitted False so that the ephemeral
    # serving tasks (whose set_caching_options(False) serializes to an empty cachingOptions) fall
    # back to off. Producers opt back in per-task at compile time, not via this submit knob.
    aip = _FakeAip()
    job = submit.submit_pipeline(_cfg(), aiplatform=aip, template_path=tmp_path / "p.yaml")
    assert job.kwargs["enable_caching"] is False


# -- preflight_automl -----------------------------------------------------------------------------
class _FakeDataset:
    def __init__(self) -> None:
        self.deleted = False

    def delete(self) -> None:
        self.deleted = True


class _PreflightAip:
    """Fake aiplatform for the AutoML preflight: optionally fails the training run."""

    def __init__(self, run_error: Exception | None = None) -> None:
        self.init_kwargs: dict | None = None
        self.dataset = _FakeDataset()
        self.job: Any = None
        outer = self

        class TimeSeriesDataset:
            @staticmethod
            def create(**_kw: object) -> _FakeDataset:
                return outer.dataset

        class AutoMLForecastingTrainingJob:
            def __init__(self, **kw: object) -> None:
                outer.job = self
                self.kwargs = kw
                self.cancelled = False
                self.deleted = False

            def run(self, **_kw: object) -> str:
                # sync=False returns immediately; any error surfaces at wait_for_resource_creation.
                return "MODEL"

            def wait_for_resource_creation(self) -> None:
                if run_error is not None:
                    raise run_error

            def cancel(self) -> None:
                self.cancelled = True

            def delete(self) -> None:
                self.deleted = True

        self.TimeSeriesDataset = TimeSeriesDataset
        self.AutoMLForecastingTrainingJob = AutoMLForecastingTrainingJob

    def init(self, **kw: object) -> None:
        self.init_kwargs = kw


def test_preflight_automl_success_cancels_job_and_deletes_dataset() -> None:
    aip = _PreflightAip()
    submit.preflight_automl(_cfg(), aiplatform=aip)
    assert aip.job.cancelled is True
    assert aip.job.deleted is True
    assert aip.dataset.deleted is True


def test_preflight_automl_propagates_error_and_still_cleans_up() -> None:
    boom = ValueError("The time column 'date' should be available at forecast.")
    aip = _PreflightAip(run_error=boom)
    with pytest.raises(ValueError, match="should be available at forecast"):
        submit.preflight_automl(_cfg(), aiplatform=aip)
    # the probe dataset is created before the failing run(), so cleanup must still fire
    assert aip.dataset.deleted is True
    assert aip.job.cancelled is True


# -- CLI ------------------------------------------------------------------------------------------
def test_cli_dry_run_compiles_without_submit(tmp_path: Path, caplog) -> None:
    out = tmp_path / "compiled.yaml"
    with caplog.at_level(logging.INFO):
        rc = submit.main(["--config", _write_config(tmp_path), "--dry-run", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert "not submitted" in " ".join(caplog.messages)


def _write_config(tmp_path: Path, config: dict | None = None) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config or BASE), encoding="utf-8")
    return str(path)


def _train_backend_executors(spec_path: Path) -> set[str]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    execs = spec["deploymentSpec"]["executors"]
    return {e for e in execs if e.startswith("exec-train-backend")}


def test_cli_disable_automl_drops_backend(tmp_path: Path) -> None:
    cfg = {
        **BASE,
        "models": [
            {"name": "timesfm", "params": {"type": "timesfm"}},
            {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
            {"name": "automl", "enabled": True, "params": {"type": "automl"}},
        ],
    }
    out = tmp_path / "compiled.yaml"
    rc = submit.main(
        [
            "--config",
            _write_config(tmp_path, cfg),
            "--disable-automl",
            "--dry-run",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    # only bqml runs in-process; the AutoML train-backend is gone (timesfm is served, not trained)
    assert _train_backend_executors(out) == {"exec-train-backend"}


def test_cli_force_data_rebuild_sets_config_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, ExperimentConfig] = {}

    def fake_submit(cfg: ExperimentConfig, **_kw: object) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr(submit, "submit_pipeline", fake_submit)
    rc = submit.main(["--config", _write_config(tmp_path), "--force-data-rebuild"])
    assert rc == 0
    assert captured["cfg"].data.force_rebuild is True


def test_cli_no_cache_disables_producer_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, ExperimentConfig] = {}

    def fake_submit(cfg: ExperimentConfig, **_kw: object) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr(submit, "submit_pipeline", fake_submit)
    rc = submit.main(["--config", _write_config(tmp_path), "--no-cache"])
    assert rc == 0
    assert captured["cfg"].pipeline.enable_caching is False


def test_cli_enable_and_disable_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        submit.main(
            [
                "--config",
                _write_config(tmp_path),
                "--enable-automl",
                "--disable-automl",
                "--dry-run",
            ]
        )
