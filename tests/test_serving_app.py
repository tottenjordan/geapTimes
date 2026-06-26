"""Offline tests for the serving Flask app (fake predictor; no model, no server)."""

from typing import Any

from geaptimes.pipelines.serving.app import create_app


class _FakePredictor:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def predict(self, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append(instances)
        return [{"series_id": inst["series_id"], "forecast": [1.0, 2.0]} for inst in instances]


def test_health_returns_200() -> None:
    app = create_app(predictor=_FakePredictor())
    resp = app.test_client().get("/health")
    assert resp.status_code == 200


def test_predict_round_trip() -> None:
    predictor = _FakePredictor()
    client = create_app(predictor=predictor).test_client()
    resp = client.post("/predict", json={"instances": [{"series_id": "a", "context": [1, 2, 3]}]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"predictions": [{"series_id": "a", "forecast": [1.0, 2.0]}]}
    assert predictor.calls == [[{"series_id": "a", "context": [1, 2, 3]}]]


def test_predict_missing_instances_is_empty() -> None:
    client = create_app(predictor=_FakePredictor()).test_client()
    resp = client.post("/predict", json={})
    assert resp.status_code == 200
    assert resp.get_json() == {"predictions": []}


def test_routes_honor_aip_env(monkeypatch) -> None:
    monkeypatch.setenv("AIP_HEALTH_ROUTE", "/healthz")
    monkeypatch.setenv("AIP_PREDICT_ROUTE", "/v1/predict")
    client = create_app(predictor=_FakePredictor()).test_client()
    assert client.get("/healthz").status_code == 200
    assert client.post("/v1/predict", json={"instances": []}).status_code == 200
