"""Offline tests: the comparison pipeline compiles to a KFP spec with the expected graph shape.

These assert the *structure* of the compiled IR (which component executors are present per config)
without any cloud calls — the compile-time branching on enabled backends, serving mode, and
teardown is what these lock down.
"""

from pathlib import Path

import yaml

from geaptimes.pipelines.compile import compile_pipeline
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "proj", "gcs_bucket": "bkt", "bq_dataset": "ds", "region": "us-central1"},
    "execution": {"experiment_name": "exp"},
}

ALL_THREE = [
    {"name": "timesfm", "params": {"type": "timesfm"}},
    {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
    {"name": "automl", "params": {"type": "automl"}},
]


def _cfg(models: list[dict], serving: dict | None = None) -> ExperimentConfig:
    over: dict = {**BASE, "models": models}
    if serving is not None:
        over["pipeline"] = {"serving": serving}
    return ExperimentConfig.model_validate(over)


def _executors(cfg: ExperimentConfig, tmp_path: Path) -> set[str]:
    out = compile_pipeline(cfg, tmp_path / "pipeline.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    return set(spec["deploymentSpec"]["executors"].keys())


def test_endpoint_mode_transient_full_graph(tmp_path: Path) -> None:
    execs = _executors(_cfg(ALL_THREE), tmp_path)
    assert "exec-build-tables" in execs
    assert "exec-register-timesfm" in execs
    assert "exec-deploy-endpoint" in execs
    assert "exec-endpoint-predict" in execs
    assert "exec-compare-backends" in execs
    # transient endpoint mode -> teardown wired as the ExitHandler exit task
    assert "exec-teardown-serving" in execs
    # batch path not present in endpoint mode
    assert "exec-batch-predict" not in execs
    # two in-process backends (bqml + automl) -> two run-backend executors
    run_backends = {e for e in execs if e.startswith("exec-run-backend")}
    assert len(run_backends) == 2


def test_batch_mode_uses_batch_predict_and_no_endpoint(tmp_path: Path) -> None:
    execs = _executors(_cfg(ALL_THREE, serving={"mode": "batch"}), tmp_path)
    assert "exec-batch-predict" in execs
    assert "exec-deploy-endpoint" not in execs
    assert "exec-endpoint-predict" not in execs
    # batch creates no endpoint, so there is nothing to tear down
    assert "exec-teardown-serving" not in execs


def test_keep_deployed_skips_teardown(tmp_path: Path) -> None:
    serving = {"mode": "endpoint", "keep_deployed": True}
    execs = _executors(_cfg(ALL_THREE, serving=serving), tmp_path)
    assert "exec-deploy-endpoint" in execs
    assert "exec-endpoint-predict" in execs
    assert "exec-teardown-serving" not in execs


def test_timesfm_disabled_skips_serving(tmp_path: Path) -> None:
    models = [
        {"name": "timesfm", "enabled": False, "params": {"type": "timesfm"}},
        {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
    ]
    execs = _executors(_cfg(models), tmp_path)
    assert "exec-run-backend" in execs
    assert "exec-register-timesfm" not in execs
    assert "exec-deploy-endpoint" not in execs
    assert "exec-batch-predict" not in execs
    assert "exec-teardown-serving" not in execs


def test_pipeline_name_and_config_param(tmp_path: Path) -> None:
    out = compile_pipeline(_cfg(ALL_THREE), tmp_path / "p.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    assert spec["pipelineInfo"]["name"] == "geaptimes-comparison-top25-h14-t14-v14"
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert "config_json" in params
