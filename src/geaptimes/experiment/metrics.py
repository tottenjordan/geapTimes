"""Minimal point-forecast metrics for experiment tracking.

Stage 3 logs a small, backend-agnostic accuracy summary (MAE / RMSE) so every tracked run is
meaningful. The point forecast is compared against the held-out TEST actuals on the shared
(series, date) keys. The full standardized evaluation suite — MAPE, quantile (pinball) loss,
per-series breakdowns, and the cross-backend comparison report — lands in Stage 5.
"""

from typing import TYPE_CHECKING

from geaptimes.models.base import DATE_COLUMN, FORECAST_COLUMN, SERIES_COLUMN

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_ACTUAL_COLUMN = "actual"


def point_metrics(
    predictions: "pd.DataFrame",
    actuals: "pd.DataFrame",
    *,
    series_col: str,
    time_col: str,
    target_col: str,
) -> dict[str, float]:
    """Return ``{mae, rmse, n_points}`` for ``predictions`` vs ``actuals``.

    ``predictions`` uses the standardized :data:`~geaptimes.models.base.PREDICTION_COLUMNS`
    (``series``/``date``/``forecast``); ``actuals`` uses the config's own column names
    (``series_col``/``time_col``/``target_col``). The two are inner-joined on (series, date), so
    only the overlapping backtest window contributes. Raises if the overlap is empty.
    """
    import numpy as np  # noqa: PLC0415 - lazy heavy import
    import pandas as pd  # noqa: PLC0415 - lazy heavy import

    preds = predictions[[SERIES_COLUMN, DATE_COLUMN, FORECAST_COLUMN]].copy()
    act = actuals[[series_col, time_col, target_col]].rename(
        columns={series_col: SERIES_COLUMN, time_col: DATE_COLUMN, target_col: _ACTUAL_COLUMN}
    )
    # Normalize join keys to plain dates so Timestamp/date representations align.
    preds[DATE_COLUMN] = pd.to_datetime(preds[DATE_COLUMN]).dt.date
    act[DATE_COLUMN] = pd.to_datetime(act[DATE_COLUMN]).dt.date

    merged = preds.merge(act, on=[SERIES_COLUMN, DATE_COLUMN], how="inner")
    if merged.empty:
        msg = "no overlapping (series, date) keys between predictions and actuals"
        raise ValueError(msg)

    error = merged[FORECAST_COLUMN].to_numpy(dtype=float) - merged[_ACTUAL_COLUMN].to_numpy(
        dtype=float
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "n_points": float(len(merged)),
    }
