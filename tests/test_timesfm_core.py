"""Offline tests for the factored TimesFM core (no real checkpoint, no torch)."""

import sys
import types

import numpy as np
import pandas as pd

from geaptimes.models.base import FORECAST_COLUMN
from geaptimes.models.timesfm_core import (
    _quantile_channels,
    compile_timesfm,
    forecast_arrays,
    rows_from_forecast,
    series_quantiles,
)

HORIZON = 14


class _FakeModel:
    """Returns deterministic point + decile arrays; records the compile config + forecast call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.compiled_with: object = None

    def compile(self, config: object) -> None:
        self.compiled_with = config

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = len(inputs)
        self.calls.append((horizon, n))
        point = np.tile(np.arange(horizon, dtype=float), (n, 1))
        # decile channels equal their index so q-channel mapping is checkable.
        quantiles = np.broadcast_to(np.arange(10, dtype=float), (n, horizon, 10)).copy()
        return point, quantiles


def test_forecast_arrays_returns_numpy() -> None:
    model = _FakeModel()
    inputs = [np.arange(50, dtype=float), np.arange(50, dtype=float)]
    point, quantiles = forecast_arrays(model, inputs, HORIZON)
    assert isinstance(point, np.ndarray)
    assert isinstance(quantiles, np.ndarray)
    assert point.shape == (2, HORIZON)
    assert quantiles.shape == (2, HORIZON, 10)
    assert model.calls == [(HORIZON, 2)]


def test_rows_from_forecast_shape_dates_and_channels() -> None:
    model = _FakeModel()
    forecast = forecast_arrays(model, [np.zeros(5), np.zeros(5)], HORIZON)
    last = pd.Timestamp("2018-05-17")
    rows = rows_from_forecast(
        ["a", "b"], [last, last], forecast, horizon=HORIZON, model_name="timesfm"
    )
    assert len(rows) == 2 * HORIZON
    first = rows[0]
    assert first["series"] == "a"
    assert first["horizon_step"] == 1
    assert first["date"] == (last + pd.Timedelta(days=1)).date()
    assert first["model"] == "timesfm"
    # decile channels equal index -> q10=1, q30=3, q50=5, q70=7, q90=9
    assert (first["q10"], first["q30"], first["q50"], first["q70"], first["q90"]) == (
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
    )


def test_series_quantiles_arrays() -> None:
    model = _FakeModel()
    point, quantiles = forecast_arrays(model, [np.zeros(5)], HORIZON)
    out = series_quantiles(point[0], quantiles[0], horizon=HORIZON)
    assert set(out) == {FORECAST_COLUMN, "q10", "q30", "q50", "q70", "q90"}
    assert len(out[FORECAST_COLUMN]) == HORIZON
    assert all(len(v) == HORIZON for v in out.values())
    assert out["q50"] == [5.0] * HORIZON
    assert out[FORECAST_COLUMN] == list(range(HORIZON))


def test_quantile_channels_helper() -> None:
    assert _quantile_channels(10) == [1, 3, 5, 7, 9]
    assert _quantile_channels(5) == [0, 1, 2, 3, 4]


def test_compile_timesfm_wires_forecast_config(monkeypatch) -> None:
    # Inject a fake `timesfm` module so we exercise the wiring without importing torch.
    captured: dict[str, object] = {}

    class _ForecastConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake = types.ModuleType("timesfm")
    setattr(fake, "ForecastConfig", _ForecastConfig)  # noqa: B010 - dynamic attr on fake module
    monkeypatch.setitem(sys.modules, "timesfm", fake)

    model = _FakeModel()
    returned = compile_timesfm(model, max_context=1024, horizon=HORIZON, quantiles=True)
    assert returned is model
    assert isinstance(model.compiled_with, _ForecastConfig)
    assert captured["max_context"] == 1024
    assert captured["max_horizon"] == HORIZON
    assert captured["use_continuous_quantile_head"] is True
    assert captured["fix_quantile_crossing"] is True
