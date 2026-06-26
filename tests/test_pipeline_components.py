"""Offline tests for the KFP component *shells* via their underlying ``python_func``.

The components run in a container in a live pipeline, but their wrapped Python function can be
called directly with the step seams monkeypatched — exercising the shell wiring (config rebuild,
delegation to ``steps``, Parquet artifact writes, the returned metrics dict) without KFP or cloud.
"""

import base64
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from geaptimes.experiment.runner import RunRecord
from geaptimes.models.base import PREDICTION_COLUMNS
from geaptimes.pipelines import steps
from geaptimes.pipelines.components import build_components
from geaptimes.pipelines.steps import BackendResult
from geaptimes.schemas import ExperimentConfig

COMPS = build_components("img:latest")

CFG = ExperimentConfig.model_validate(
    {
        "project": {"id": "p", "gcs_bucket": "b", "bq_dataset": "ds", "region": "us-central1"},
        "forecast": {"horizon": 2},
        "models": [
            {"name": "timesfm", "params": {"type": "timesfm"}},
            {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
        ],
        "execution": {"experiment_name": "exp"},
    }
)
CFG_JSON = CFG.model_dump_json()


class _FakeArtifact:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeTracker:
    def __init__(self, *_a: object, **_k: object) -> None:
        pass


def _predictions() -> pd.DataFrame:
    rows = [
        {
            "series": "s0",
            "date": (pd.Timestamp("2018-03-18") + pd.Timedelta(days=s)).date(),
            "horizon_step": s + 1,
            "forecast": 10.0 + s,
            "q10": 9.0,
            "q30": 9.5,
            "q50": 10.0,
            "q70": 10.5,
            "q90": 11.0,
            "model": "m",
        }
        for s in range(2)
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _backend_result(model: str) -> BackendResult:
    record = RunRecord(
        run_name="r", model=model, overrides={}, metrics={"mae": 1.5, "rmse": 2.5, "n_points": 2}
    )
    return BackendResult(record=record, predictions=_predictions())


class _FakeDatasetArtifact:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.uri = ""
        self.metadata: dict[str, Any] = {}


def test_build_tables_component(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        steps,
        "build_tables_step",
        lambda _cfg: {
            "prepped": steps.TableRef("p.ds.prepped__x", 100),
            "train": steps.TableRef("p.ds.train__x", 80),
            "infer": steps.TableRef("p.ds.infer__x", 20),
        },
    )
    prepped, train, infer = _FakeDatasetArtifact(), _FakeDatasetArtifact(), _FakeDatasetArtifact()
    out = COMPS.build_tables.python_func(
        config_json=CFG_JSON, prepped=prepped, train=train, infer=infer
    )
    assert out is None  # build_tables now emits artifacts, no return value
    assert infer.uri == "bq://p.ds.infer__x"
    assert train.metadata["rows"] == 80
    assert prepped.metadata["table"] == "p.ds.prepped__x"
    # Every table artifact carries the same config fingerprint (a non-empty hash).
    assert prepped.metadata["fingerprint"] == infer.metadata["fingerprint"]
    assert prepped.metadata["fingerprint"]


def test_train_backend_component(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_train(_cfg: object, model_name: str) -> str:
        seen["model_name"] = model_name
        return f"ref-{model_name}"

    monkeypatch.setattr(steps, "train_backend_step", fake_train)
    out = COMPS.train_backend.python_func(
        config_json=CFG_JSON, model_name="bqml_arima_xreg", train=_FakeDatasetArtifact()
    )
    assert out == "ref-bqml_arima_xreg"
    assert seen["model_name"] == "bqml_arima_xreg"


def test_infer_backend_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_infer(_cfg: object, model_name: str, model_reference: str) -> pd.DataFrame:
        seen["model_name"] = model_name
        seen["model_reference"] = model_reference
        return _predictions()

    monkeypatch.setattr(steps, "infer_backend_step", fake_infer)
    art = _FakeArtifact(str(tmp_path / "preds.parquet"))
    out = COMPS.infer_backend.python_func(
        config_json=CFG_JSON,
        model_name="bqml_arima_xreg",
        model_reference="ref-bqml_arima_xreg",
        infer=_FakeDatasetArtifact(),
        predictions=art,
    )
    assert out is None  # infer writes the predictions artifact; scoring happens downstream
    assert seen["model_reference"] == "ref-bqml_arima_xreg"
    assert len(pd.read_parquet(art.path)) == 2


def test_score_and_track_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_score(
        _cfg: object, model_name: str, _frame: object, *, tracker: object
    ) -> BackendResult:
        seen["model_name"] = model_name
        seen["tracker"] = tracker
        return _backend_result(model_name)

    monkeypatch.setattr(steps, "score_and_track_step", fake_score)
    monkeypatch.setattr("geaptimes.experiment.tracking.ExperimentTracker", _FakeTracker)
    art = _FakeArtifact(str(tmp_path / "preds.parquet"))
    _predictions().to_parquet(art.path, index=False)  # infer step's artifact, read back here
    out = COMPS.score_and_track.python_func(
        config_json=CFG_JSON, model_name="bqml_arima_xreg", predictions=art
    )
    # Substitution-safe contract: score_and_track outputs are collected into compare_backends'
    # ``rows`` list, which KFP builds by *textual* substitution into the executor-input JSON. The
    # output must therefore contain no character (``"`` ``,`` ``]``) that could break that JSON --
    # i.e. it must stay within the base64 alphabet. (Regressing to raw JSON re-breaks the live run.)
    assert re.fullmatch(r"[A-Za-z0-9+/=]+", out)
    assert json.loads(base64.b64decode(out)) == {
        "model": "bqml_arima_xreg",
        "mae": 1.5,
        "rmse": 2.5,
    }
    assert seen["model_name"] == "bqml_arima_xreg"
    assert isinstance(seen["tracker"], _FakeTracker)


def test_register_and_deploy_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(steps, "register_timesfm_step", lambda _cfg: "projects/x/models/m")
    monkeypatch.setattr(steps, "deploy_endpoint_step", lambda _cfg, name: f"ep-for-{name}")
    registered = COMPS.register_timesfm.python_func(
        config_json=CFG_JSON, prepped=_FakeDatasetArtifact()
    )
    assert registered == "projects/x/models/m"
    deployed = COMPS.deploy_endpoint.python_func(config_json=CFG_JSON, model_resource_name="m")
    assert deployed == "ep-for-m"


def test_endpoint_predict_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The serving predict component now only emits the predictions artifact; scoring/tracking is
    # the downstream shared score_and_track step.
    monkeypatch.setattr(steps, "endpoint_predict_step", lambda _cfg, _name: _predictions())
    art = _FakeArtifact(str(tmp_path / "tfm.parquet"))
    out = COMPS.endpoint_predict.python_func(
        config_json=CFG_JSON, endpoint_resource_name="ep", predictions=art
    )
    assert out is None
    assert len(pd.read_parquet(art.path)) == 2


def test_batch_predict_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(steps, "batch_predict_timesfm_step", lambda _cfg, _name: _predictions())
    art = _FakeArtifact(str(tmp_path / "tfm.parquet"))
    out = COMPS.batch_predict.python_func(
        config_json=CFG_JSON, model_resource_name="m", predictions=art
    )
    assert out is None
    assert len(pd.read_parquet(art.path)) == 2


def test_teardown_serving_component(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[object] = []
    monkeypatch.setattr(steps, "teardown_serving_step", called.append)
    COMPS.teardown_serving.python_func(config_json=CFG_JSON)
    assert len(called) == 1


def test_compare_backends_component(tmp_path: Path) -> None:
    art = _FakeArtifact(str(tmp_path / "comparison.json"))
    rows = [
        base64.b64encode(json.dumps({"model": "timesfm", "mae": 80.0, "rmse": 110.0}).encode()),
        base64.b64encode(
            json.dumps({"model": "bqml_arima_xreg", "mae": 77.0, "rmse": 101.0}).encode()
        ),
    ]
    winner = COMPS.compare_backends.python_func(rows=rows, comparison=art)
    assert winner == "bqml_arima_xreg"
    saved = json.loads(Path(art.path).read_text(encoding="utf-8"))
    assert saved["winner"] == "bqml_arima_xreg"
    assert [r["model"] for r in saved["ranking"]] == ["bqml_arima_xreg", "timesfm"]
