"""Offline tests for the TimesFM serving predictor (fake model; no torch, no checkpoint)."""

import numpy as np

from geaptimes.pipelines.serving.predictor import PredictorConfig, TimesFMPredictor

HORIZON = 7


class _FakeModel:
    """Deterministic point + decile arrays; records contexts it was forecast with."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []
        self.load_count = 0

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        self.seen = list(inputs)
        n = len(inputs)
        point = np.tile(np.arange(horizon, dtype=float), (n, 1))
        quantiles = np.broadcast_to(np.arange(10, dtype=float), (n, horizon, 10)).copy()
        return point, quantiles


def _predictor(model: _FakeModel, *, context_len: int = 512) -> TimesFMPredictor:
    config = PredictorConfig(horizon=HORIZON, context_len=context_len)

    def loader(_config: PredictorConfig) -> _FakeModel:
        model.load_count += 1
        return model

    return TimesFMPredictor(config, model_loader=loader)


def test_predict_response_shape_and_quantiles() -> None:
    model = _FakeModel()
    instances = [
        {"series_id": "a", "context": list(range(600))},
        {"series_id": "b", "context": list(range(600))},
    ]
    responses = _predictor(model).predict(instances)

    assert [r["series_id"] for r in responses] == ["a", "b"]
    first = responses[0]
    assert first["horizon"] == HORIZON
    assert len(first["forecast"]) == HORIZON
    assert set(first) == {"series_id", "horizon", "forecast", "q10", "q30", "q50", "q70", "q90"}
    # decile channels equal their index -> q50 == channel 5
    assert first["q50"] == [5.0] * HORIZON
    assert first["forecast"] == list(range(HORIZON))


def test_predict_slices_context_to_context_len() -> None:
    model = _FakeModel()
    _predictor(model, context_len=128).predict([{"series_id": "a", "context": list(range(600))}])
    assert model.seen[0].shape == (128,)
    # the last context_len points are kept
    assert model.seen[0][-1] == 599.0


def test_model_loaded_once_and_cached() -> None:
    model = _FakeModel()
    predictor = _predictor(model)
    predictor.predict([{"series_id": "a", "context": [1.0, 2.0]}])
    predictor.predict([{"series_id": "a", "context": [1.0, 2.0]}])
    assert model.load_count == 1


def test_empty_instances_returns_empty() -> None:
    model = _FakeModel()
    assert _predictor(model).predict([]) == []
    assert model.load_count == 0  # no model load when there's nothing to do


def test_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TIMESFM_HORIZON", "21")
    monkeypatch.setenv("TIMESFM_CONTEXT_LEN", "256")
    monkeypatch.setenv("TIMESFM_QUANTILES", "false")
    config = PredictorConfig.from_env()
    assert config.horizon == 21
    assert config.context_len == 256
    assert config.quantiles is False
