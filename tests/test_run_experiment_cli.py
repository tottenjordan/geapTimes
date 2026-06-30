"""Offline tests for the run_experiment CLI (fake runner; no cloud, no models)."""

from pathlib import Path

import pytest

from geaptimes.experiment import cli
from geaptimes.experiment.runner import RunRecord
from geaptimes.schemas import ExperimentConfig

_CONFIG_YAML = """
project:
  id: p
  gcs_bucket: b
  region: us-central1
models:
  - name: timesfm
    params: {type: timesfm}
  - name: automl
    enabled: false
    params: {type: automl}
execution:
  experiment_name: exp
"""


def _write_config(tmp_path: Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG_YAML, encoding="utf-8")
    return str(path)


def _record(model: str, mae: float, rmse: float) -> RunRecord:
    return RunRecord(
        run_name=f"{model}-base-20260101-000000",
        model=model,
        overrides={},
        metrics={"mae": mae, "rmse": rmse, "smape": 10.0, "quantile_loss": 1.0, "n_points": 14.0},
    )


# -- pure helpers ---------------------------------------------------------------------------------


def test_planned_runs_lists_enabled_models_only() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "project": {"id": "p", "gcs_bucket": "b", "region": "us-central1"},
            "models": [
                {"name": "timesfm", "params": {"type": "timesfm"}},
                {"name": "automl", "enabled": False, "params": {"type": "automl"}},
            ],
            "execution": {"experiment_name": "exp"},
        }
    )
    runs = cli.planned_runs(cfg)
    assert runs == [("base", "timesfm")]  # automl disabled -> excluded


def test_format_records_table_has_header_and_rows() -> None:
    table = cli.format_records_table([_record("timesfm", 85.5, 113.6)])
    lines = table.splitlines()
    assert lines[0].split() == ["run", "model", "mae", "rmse", "smape", "quantile_loss"]
    assert "timesfm" in lines[2]
    assert "85.5000" in lines[2]


def test_build_comparison_report_ranks_records() -> None:
    records = [_record("timesfm", 85.5, 113.6), _record("bqml_arima_xreg", 77.7, 101.7)]
    report = cli.build_comparison_report(records)
    assert "bqml_arima_xreg (winner)" in report  # lower RMSE wins


# -- main ------------------------------------------------------------------------------------------


def test_main_runs_and_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path)
    out = tmp_path / "report.md"
    records = [_record("timesfm", 85.5, 113.6), _record("bqml_arima_xreg", 77.7, 101.7)]
    rc = cli.main(
        ["--config", config, "--output", str(out)],
        runner=lambda _cfg: records,
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "timesfm" in printed
    assert "(winner)" in printed
    assert out.read_text(encoding="utf-8").startswith("| rank | model |")


def test_main_dry_run_does_not_invoke_runner(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    def _boom(_cfg: ExperimentConfig) -> list[RunRecord]:
        msg = "runner must not be called on --dry-run"
        raise AssertionError(msg)

    assert cli.main(["--config", config, "--dry-run"], runner=_boom) == 0


def test_main_enable_automl_override_reaches_runner(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    seen: dict[str, ExperimentConfig] = {}

    def _capture(cfg: ExperimentConfig) -> list[RunRecord]:
        seen["cfg"] = cfg
        return []

    cli.main(["--config", config, "--enable-automl"], runner=_capture)
    automl = next(m for m in seen["cfg"].models if m.params.type == "automl")
    assert automl.enabled is True


def test_main_disable_automl_override_reaches_runner(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    seen: dict[str, ExperimentConfig] = {}

    def _capture(cfg: ExperimentConfig) -> list[RunRecord]:
        seen["cfg"] = cfg
        return []

    cli.main(["--config", config, "--disable-automl"], runner=_capture)
    automl = next(m for m in seen["cfg"].models if m.params.type == "automl")
    assert automl.enabled is False
