"""Offline tests for TimesFMForecaster (no checkpoint download, no BigQuery)."""

import numpy as np
import pandas as pd

from geaptimes.models.base import PREDICTION_COLUMNS
from geaptimes.models.timesfm import TimesFMForecaster, _quantile_channels
from geaptimes.schemas import ExperimentConfig

HORIZON = 14
CONTEXT = 512

BASE = {
    "project": {"id": "p", "gcs_bucket": "b"},
    "forecast": {"horizon": HORIZON},
    "models": [{"name": "timesfm", "params": {"type": "timesfm", "context_len": CONTEXT}}],
    "execution": {"experiment_name": "exp"},
}


def _cfg(**over: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **over})


def _fake_frame(n_series: int = 2, days: int = 600) -> pd.DataFrame:
    """Long frame with TRAIN/VALIDATE/TEST splits for a few series."""
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    frames = []
    for s in range(n_series):
        df = pd.DataFrame(
            {
                "start_station_name": f"station_{s}",
                "date": [d.date() for d in dates],
                "num_trips": np.arange(days) + s,
            }
        )
        df["splits"] = "TRAIN"
        df.loc[df.index[-HORIZON:], "splits"] = "TEST"
        df.loc[df.index[-2 * HORIZON : -HORIZON], "splits"] = "VALIDATE"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class _FakeModel:
    """Records the forecast call and returns deterministic point + decile arrays."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = len(inputs)
        self.calls.append((horizon, n))
        point = np.tile(np.arange(horizon, dtype=float), (n, 1))
        # 10 decile channels: channel c == c so q-channel mapping is checkable.
        quantiles = np.broadcast_to(np.arange(10, dtype=float), (n, horizon, 10)).copy()
        return point, quantiles


def _forecaster(frame: pd.DataFrame, model: _FakeModel, **over: object) -> TimesFMForecaster:
    cfg = _cfg(**over)
    return TimesFMForecaster(
        cfg.models[0],
        cfg,
        model_loader=lambda _checkpoint: model,
        frame_loader=lambda: frame,
        cuda_available=lambda: False,
    )


def test_predict_output_schema_and_horizon() -> None:
    frame = _fake_frame(n_series=2)
    model = _FakeModel()
    result = _forecaster(frame, model).predict()

    assert list(result.predictions.columns) == PREDICTION_COLUMNS
    assert len(result.predictions) == 2 * HORIZON  # 2 series x horizon
    assert model.calls == [(HORIZON, 2)]
    assert result.metadata["n_series"] == 2
    assert result.metadata["context_len"] == CONTEXT


def test_context_truncation_and_train_only() -> None:
    # 600 days of history but context_len=512 → only the last 512 train points are fed.
    frame = _fake_frame(n_series=1, days=600)
    captured: dict[str, np.ndarray] = {}

    class CapModel(_FakeModel):
        def forecast(self, horizon: int, inputs: list[np.ndarray]):  # type: ignore[override]
            captured["arr"] = inputs[0]
            return super().forecast(horizon, inputs)

    _forecaster(frame, CapModel()).predict()
    # 600 total - 14 TEST excluded = 586 train rows, truncated to context_len.
    assert captured["arr"].shape == (CONTEXT,)


def test_quantile_channel_mapping() -> None:
    frame = _fake_frame(n_series=1)
    result = _forecaster(frame, _FakeModel()).predict()
    row = result.predictions.iloc[0]
    # decile channels equal their index → q10=1, q30=3, q50=5, q70=7, q90=9
    assert (row["q10"], row["q30"], row["q50"], row["q70"], row["q90"]) == (1.0, 3.0, 5.0, 7.0, 9.0)


def test_forecast_dates_follow_last_train_date() -> None:
    frame = _fake_frame(n_series=1)
    last_train = frame[frame["splits"] != "TEST"]["date"].max()
    result = _forecaster(frame, _FakeModel()).predict()
    first = result.predictions.sort_values("horizon_step").iloc[0]
    assert first["date"] == last_train + pd.Timedelta(days=1)
    assert first["horizon_step"] == 1


def test_device_resolution() -> None:
    frame = _fake_frame(n_series=1)
    model = _FakeModel()
    cpu = _forecaster(frame, model, execution={"experiment_name": "e", "target": "local_cpu"})
    assert cpu.device == "cpu"

    gpu_no_cuda = _forecaster(
        frame, model, execution={"experiment_name": "e", "target": "local_gpu"}
    )
    assert gpu_no_cuda.device == "cpu"  # cuda_available=False → fallback

    cfg = _cfg(execution={"experiment_name": "e", "target": "local_gpu"})
    gpu = TimesFMForecaster(
        cfg.models[0],
        cfg,
        model_loader=lambda _c: model,
        frame_loader=lambda: frame,
        cuda_available=lambda: True,
    )
    assert gpu.device == "cuda"


def test_quantile_channels_helper() -> None:
    assert _quantile_channels(10) == [1, 3, 5, 7, 9]
    assert _quantile_channels(5) == [0, 1, 2, 3, 4]
