"""Offline tests for ExperimentTracker (fake aiplatform + recording sink; no cloud)."""

import pandas as pd
import pytest

from geaptimes.experiment.tracking import ExperimentTracker, _coerce_params
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "p", "gcs_bucket": "b", "region": "us-central1"},
    "models": [{"name": "timesfm", "params": {"type": "timesfm"}}],
    "execution": {"experiment_name": "exp"},
}


def _cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(BASE)


class FakeAip:
    """Records the aiplatform experiment-API calls the tracker makes."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.params: list[dict] = []
        self.metrics: list[dict] = []

    def init(self, **kw: object) -> None:
        self.calls.append(("init", kw))

    def start_run(self, name: str) -> None:
        self.calls.append(("start_run", name))

    def end_run(self) -> None:
        self.calls.append(("end_run",))

    def log_params(self, params: dict) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.append(metrics)


def _tracker() -> tuple[ExperimentTracker, FakeAip, list[tuple[str, str]]]:
    aip = FakeAip()
    sink_writes: list[tuple[str, str]] = []
    tracker = ExperimentTracker(
        _cfg(), aiplatform=aip, artifact_sink=lambda uri, text: sink_writes.append((uri, text))
    )
    return tracker, aip, sink_writes


def test_run_lifecycle_order() -> None:
    tracker, aip, _ = _tracker()
    with tracker.run("run-1"):
        pass
    assert aip.calls[0][0] == "init"
    assert aip.calls[0][1]["experiment"] == "exp"
    assert aip.calls[0][1]["staging_bucket"] == "gs://b"
    assert aip.calls[1] == ("start_run", "run-1")
    assert aip.calls[2] == ("end_run",)


def test_end_run_called_on_exception() -> None:
    tracker, aip, _ = _tracker()
    with pytest.raises(RuntimeError), tracker.run("run-x"):
        raise RuntimeError
    assert aip.calls[-1] == ("end_run",)


def test_log_params_coerces_and_forwards() -> None:
    tracker, aip, _ = _tracker()
    tracker.log_params({"model": "timesfm", "horizon": 14, "axes": {"a": 1}})
    assert aip.params[0]["model"] == "timesfm"
    assert aip.params[0]["horizon"] == 14
    assert aip.params[0]["axes"] == '{"a": 1}'  # non-primitive -> json string


def test_log_metrics_casts_to_float() -> None:
    tracker, aip, _ = _tracker()
    tracker.log_metrics({"mae": 1, "rmse": 2.5})
    assert aip.metrics[0] == {"mae": 1.0, "rmse": 2.5}


def test_artifact_base_uri() -> None:
    tracker, _, _ = _tracker()
    assert tracker.artifact_base("run-1") == "gs://b/experiments/exp/run-1"


def test_log_artifacts_writes_config_and_predictions() -> None:
    tracker, _, writes = _tracker()
    preds = pd.DataFrame({"series": ["s"], "forecast": [1.0]})
    base = tracker.log_artifacts("run-1", config=_cfg(), predictions=preds)
    assert base == "gs://b/experiments/exp/run-1"
    uris = [uri for uri, _ in writes]
    assert uris == [
        "gs://b/experiments/exp/run-1/config.json",
        "gs://b/experiments/exp/run-1/predictions.csv",
    ]
    assert "series,forecast" in writes[1][1]


def test_coerce_params_helper() -> None:
    out = _coerce_params({"s": "x", "i": 3, "f": 1.5, "b": True, "list": [1, 2]})
    assert out == {"s": "x", "i": 3, "f": 1.5, "b": True, "list": "[1, 2]"}
