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


class _FakeMetrics:
    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def log_metric(self, name: str, value: float) -> None:
        self.scalars[name] = value


def test_ensure_source_component(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(_cfg: object, *, force: bool) -> steps.TableRef:
        seen["force"] = force
        return steps.TableRef("p.ds.src__x", 44)

    monkeypatch.setattr(steps, "ensure_source_step", fake)
    source = _FakeDatasetArtifact()
    out = COMPS.ensure_source.python_func(config_json=CFG_JSON, source=source)
    assert out is None  # emits a lineage artifact, no return value
    assert source.uri == "bq://p.ds.src__x"
    assert source.metadata["table"] == "p.ds.src__x"
    assert source.metadata["rows"] == 44
    assert source.metadata["fingerprint"]  # a non-empty config fingerprint
    assert seen["force"] is False  # threaded from cfg.data.force_rebuild (default False)


def test_ensure_prepped_component(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(_cfg: object, *, force: bool) -> steps.TableRef:
        seen["force"] = force
        return steps.TableRef("p.ds.prepped__x", 90)

    monkeypatch.setattr(steps, "ensure_prepped_step", fake)
    prepped = _FakeDatasetArtifact()
    out = COMPS.ensure_prepped.python_func(
        config_json=CFG_JSON, source=_FakeDatasetArtifact(), prepped=prepped
    )
    assert out is None
    assert prepped.uri == "bq://p.ds.prepped__x"
    assert prepped.metadata["rows"] == 90
    assert prepped.metadata["fingerprint"]
    assert seen["force"] is False  # threaded from cfg.data.force_rebuild (default False)


def test_ensure_source_component_forwards_force_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(_cfg: object, *, force: bool) -> steps.TableRef:
        seen["force"] = force
        return steps.TableRef("p.ds.src__x", 1)

    monkeypatch.setattr(steps, "ensure_source_step", fake)
    forced = CFG.model_copy(update={"data": CFG.data.model_copy(update={"force_rebuild": True})})
    COMPS.ensure_source.python_func(
        config_json=forced.model_dump_json(), source=_FakeDatasetArtifact()
    )
    assert seen["force"] is True


def test_build_tables_component(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        steps,
        "build_tables_step",
        lambda _cfg: {
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
    assert train.metadata["table"] == "p.ds.train__x"
    # train + infer carry the same config fingerprint (a non-empty hash); prepped is an *input*.
    assert train.metadata["fingerprint"] == infer.metadata["fingerprint"]
    assert train.metadata["fingerprint"]


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
    metrics = _FakeMetrics()
    out = COMPS.score_and_track.python_func(
        config_json=CFG_JSON, model_name="bqml_arima_xreg", predictions=art, metrics=metrics
    )
    # MAE/RMSE are logged as scalar Metrics for the Vertex Pipelines UI (4A.4 richer artifacts).
    assert metrics.scalars == {"mae": 1.5, "rmse": 2.5}
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


def test_endpoint_predict_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The serving predict component reads the Vertex endpoint resource name from the GCPC
    # VertexEndpoint artifact's metadata, then only emits the predictions artifact; scoring/tracking
    # is the downstream shared score_and_track step.
    seen: dict[str, Any] = {}

    def fake_predict(_cfg: object, endpoint_resource_name: str) -> pd.DataFrame:
        seen["endpoint"] = endpoint_resource_name
        return _predictions()

    monkeypatch.setattr(steps, "endpoint_predict_step", fake_predict)
    endpoint = _FakeDatasetArtifact()
    endpoint.metadata["resourceName"] = "projects/x/endpoints/e"
    art = _FakeArtifact(str(tmp_path / "tfm.parquet"))
    out = COMPS.endpoint_predict.python_func(
        config_json=CFG_JSON, endpoint=endpoint, prepped=_FakeDatasetArtifact(), predictions=art
    )
    assert out is None
    assert seen["endpoint"] == "projects/x/endpoints/e"
    assert len(pd.read_parquet(art.path)) == 2


def test_batch_predict_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_batch(_cfg: object, model_resource_name: str) -> pd.DataFrame:
        seen["model"] = model_resource_name
        return _predictions()

    monkeypatch.setattr(steps, "batch_predict_timesfm_step", fake_batch)
    model = _FakeDatasetArtifact()
    model.metadata["resourceName"] = "projects/x/models/m"
    art = _FakeArtifact(str(tmp_path / "tfm.parquet"))
    out = COMPS.batch_predict.python_func(
        config_json=CFG_JSON, model=model, prepped=_FakeDatasetArtifact(), predictions=art
    )
    assert out is None
    assert seen["model"] == "projects/x/models/m"
    assert len(pd.read_parquet(art.path)) == 2


def test_teardown_serving_component(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[object] = []
    monkeypatch.setattr(steps, "teardown_serving_step", called.append)
    COMPS.teardown_serving.python_func(config_json=CFG_JSON)
    assert len(called) == 1


def test_compare_backends_component(tmp_path: Path) -> None:
    art = _FakeArtifact(str(tmp_path / "comparison.json"))
    md = _FakeArtifact(str(tmp_path / "ranking.md"))
    rows = [
        base64.b64encode(json.dumps({"model": "timesfm", "mae": 80.0, "rmse": 110.0}).encode()),
        base64.b64encode(
            json.dumps({"model": "bqml_arima_xreg", "mae": 77.0, "rmse": 101.0}).encode()
        ),
    ]
    winner = COMPS.compare_backends.python_func(rows=rows, comparison=art, ranking_md=md)
    assert winner == "bqml_arima_xreg"
    saved = json.loads(Path(art.path).read_text(encoding="utf-8"))
    assert saved["winner"] == "bqml_arima_xreg"
    assert [r["model"] for r in saved["ranking"]] == ["bqml_arima_xreg", "timesfm"]
    # The Markdown ranking artifact (4A.4): a table with the winner flagged, lowest RMSE first.
    md_text = Path(md.path).read_text(encoding="utf-8")
    assert "| rank | model | mae | rmse |" in md_text
    assert "bqml_arima_xreg (winner)" in md_text
    body = [line for line in md_text.splitlines() if line.startswith("| 1 ")]
    assert "bqml_arima_xreg" in body[0]
