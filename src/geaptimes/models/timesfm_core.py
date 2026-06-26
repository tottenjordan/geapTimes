"""TimesFM forecasting core — the model-only logic shared by the in-process forecaster and the
custom serving container.

These helpers deliberately know nothing about BigQuery, config objects, or the experiment harness:
they take a loaded model + raw context arrays and return either the standardized long rows
(:func:`rows_from_forecast`, used by :class:`~geaptimes.models.timesfm.TimesFMForecaster`) or
per-series forecast arrays (:func:`series_quantiles`, used by the serving predictor). Heavy imports
(``timesfm``/``numpy``/``pandas``) stay lazy so importing this module is cheap.
"""

from typing import TYPE_CHECKING, Any

from geaptimes.models.base import (
    DATE_COLUMN,
    FORECAST_COLUMN,
    HORIZON_STEP_COLUMN,
    MODEL_COLUMN,
    QUANTILES,
    SERIES_COLUMN,
    quantile_column,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# TimesFM's continuous-quantile head emits 10 channels: index 0 is the mean, indices 1..9 are
# the deciles q10..q90. So quantile level ``q`` lives at channel ``round(q * 10)``.
_DECILE_CHANNELS = 10


def _quantile_channels(n_channels: int) -> list[int]:
    """Channel index per :data:`QUANTILES` level for the model's quantile output."""
    if n_channels >= _DECILE_CHANNELS:
        return [round(q * 10) for q in QUANTILES]
    # Fallback: a head that emits exactly our levels positionally.
    return list(range(len(QUANTILES)))


def compile_timesfm(
    model: Any,  # noqa: ANN401 - external model
    *,
    max_context: int,
    horizon: int,
    quantiles: bool,
) -> Any:  # noqa: ANN401 - external model
    """Compile a loaded TimesFM 2.5 model with the project's forecast config; return it."""
    import timesfm  # noqa: PLC0415 - heavy optional import, kept lazy

    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=quantiles,
            fix_quantile_crossing=quantiles,
        )
    )
    return model


def forecast_arrays(
    model: Any,  # noqa: ANN401 - external model
    inputs: "list[np.ndarray]",
    horizon: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Run the model forecast and return ``(point, quantiles)`` as numpy arrays.

    ``point`` has shape ``(n_series, horizon)``; ``quantiles`` has shape
    ``(n_series, horizon, n_channels)``.
    """
    import numpy as np  # noqa: PLC0415 - lazy heavy import

    point, quantiles = model.forecast(horizon, inputs=inputs)
    return np.asarray(point), np.asarray(quantiles)


def rows_from_forecast(
    series_ids: "list[Any]",
    last_dates: "list[Any]",
    forecast: "tuple[np.ndarray, np.ndarray]",
    *,
    horizon: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """Map model outputs onto standardized long rows (one per series x horizon step).

    ``forecast`` is the ``(point, quantiles)`` pair from :func:`forecast_arrays`.
    """
    import pandas as pd  # noqa: PLC0415 - lazy heavy import

    point, quantiles = forecast
    channels = _quantile_channels(quantiles.shape[-1])
    rows: list[dict[str, Any]] = []
    for i, series_id in enumerate(series_ids):
        for step in range(horizon):
            row: dict[str, Any] = {
                SERIES_COLUMN: series_id,
                DATE_COLUMN: (last_dates[i] + pd.Timedelta(days=step + 1)).date(),
                HORIZON_STEP_COLUMN: step + 1,
                FORECAST_COLUMN: float(point[i, step]),
                MODEL_COLUMN: model_name,
            }
            for level, channel in zip(QUANTILES, channels, strict=True):
                row[quantile_column(level)] = float(quantiles[i, step, channel])
            rows.append(row)
    return rows


def series_quantiles(
    point_row: "np.ndarray",
    quantiles_row: "np.ndarray",
    *,
    horizon: int,
) -> dict[str, list[float]]:
    """Map one series' outputs to serving arrays: ``{"point": [...], "q10": [...], ...}``.

    ``point_row`` has shape ``(horizon,)``; ``quantiles_row`` has shape ``(horizon, n_channels)``.
    """
    channels = _quantile_channels(quantiles_row.shape[-1])
    out: dict[str, list[float]] = {
        FORECAST_COLUMN: [float(point_row[step]) for step in range(horizon)]
    }
    for level, channel in zip(QUANTILES, channels, strict=True):
        out[quantile_column(level)] = [
            float(quantiles_row[step, channel]) for step in range(horizon)
        ]
    return out
