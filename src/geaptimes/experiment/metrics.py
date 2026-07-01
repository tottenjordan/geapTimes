"""Standardized forecast-evaluation metrics.

Stage 3 shipped :func:`point_metrics` — a small, backend-agnostic accuracy summary (MAE / RMSE) so
every tracked run was meaningful. **Stage 5** adds the full standardized suite via :func:`evaluate`:
symmetric MAPE (sMAPE), quantile (pinball) loss across the standardized deciles, and a per-series
breakdown — all over the same identical backtest (the point forecast and quantiles compared against
the held-out TEST actuals on the shared (series, date) keys). One implementation feeds both the
offline runner and the pipeline scorer, so "identical backtests" is enforced by construction.

sMAPE (not MAPE) is used because the targets are daily per-station trip counts with zero-fill: raw
MAPE divides by ~0 on low-count stations. sMAPE's symmetric denominator ``(|a| + |f|) / 2`` is
guarded so a point where both actual and forecast are zero (a perfect prediction) contributes 0
rather than NaN. sMAPE is reported as a percentage in ``[0, 200]``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from geaptimes.models.base import (
    DATE_COLUMN,
    FORECAST_COLUMN,
    QUANTILES,
    SERIES_COLUMN,
    quantile_column,
)
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = get_logger(__name__)

_ACTUAL_COLUMN = "actual"


@dataclass(frozen=True)
class Evaluation:
    """The full standardized evaluation of one backend over one backtest.

    Aggregate scalars (``mae``/``rmse`` match :func:`point_metrics` exactly) plus a ``per_series``
    breakdown (one dict per series with ``series`` + the same scalar metrics). :meth:`summary`
    returns just the aggregate scalars, ready to hand to ``ExperimentTracker.log_metrics``.
    """

    mae: float
    rmse: float
    smape: float
    quantile_loss: float
    pmae: float
    prmse: float
    n_points: float
    per_series: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Aggregate scalar metrics only (no per-series breakdown)."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "smape": self.smape,
            "quantile_loss": self.quantile_loss,
            "pmae": self.pmae,
            "prmse": self.prmse,
            "n_points": self.n_points,
        }


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

    This is the Stage 3 minimal summary, retained for back-compat; :func:`evaluate` returns the full
    Stage 5 suite (sMAPE + quantile loss + per-series) as a superset.
    """
    import numpy as np  # noqa: PLC0415 - lazy heavy import

    merged = _merge(
        predictions,
        actuals,
        series_col=series_col,
        time_col=time_col,
        target_col=target_col,
        value_cols=[FORECAST_COLUMN],
    )
    error = merged[FORECAST_COLUMN].to_numpy(dtype=float) - merged[_ACTUAL_COLUMN].to_numpy(
        dtype=float
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "n_points": float(len(merged)),
    }


def evaluate(  # noqa: PLR0913 - predictions + actuals + 3 config column names + quantiles knob
    predictions: "pd.DataFrame",
    actuals: "pd.DataFrame",
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    quantiles: "list[float] | None" = None,
) -> Evaluation:
    """Return the full standardized :class:`Evaluation` for ``predictions`` vs ``actuals``.

    Computes aggregate ``mae``, ``rmse``, ``smape`` (percent, guarded), ``quantile_loss`` (mean
    pinball loss across ``quantiles``), the demand-normalized ``pmae``/``prmse``, and ``n_points`` —
    plus a ``per_series`` breakdown (the same metrics grouped by series, sorted by series name).
    ``mae``/``rmse``/``n_points`` match :func:`point_metrics` exactly. ``predictions`` must carry
    the standardized ``qXX`` quantile columns for the requested ``quantiles`` (every backend emits
    them). Inner-joined on (series, date); raises if the overlap is empty.
    """
    levels = QUANTILES if quantiles is None else quantiles
    value_cols = [FORECAST_COLUMN, *(quantile_column(q) for q in levels)]
    merged = _merge(
        predictions,
        actuals,
        series_col=series_col,
        time_col=time_col,
        target_col=target_col,
        value_cols=value_cols,
    )

    per_series = [
        {"series": series, **_summary(group, levels)}
        for series, group in merged.groupby(SERIES_COLUMN, sort=True)
    ]
    return Evaluation(**_summary(merged, levels), per_series=per_series)


def _merge(  # noqa: PLR0913 - predictions + actuals + 3 config column names + value_cols selector
    predictions: "pd.DataFrame",
    actuals: "pd.DataFrame",
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    value_cols: list[str],
) -> "pd.DataFrame":
    """Inner-join standardized predictions to config-named actuals on (series, date).

    Selects ``[series, date, *value_cols]`` from ``predictions`` and renames the actuals' columns to
    the standardized keys + ``actual``. Join keys are normalized to plain dates so Timestamp/date
    representations align. Raises ``ValueError`` if no (series, date) keys overlap.
    """
    import pandas as pd  # noqa: PLC0415 - lazy heavy import

    preds = predictions[[SERIES_COLUMN, DATE_COLUMN, *value_cols]].copy()
    act = actuals[[series_col, time_col, target_col]].rename(
        columns={series_col: SERIES_COLUMN, time_col: DATE_COLUMN, target_col: _ACTUAL_COLUMN}
    )
    preds[DATE_COLUMN] = pd.to_datetime(preds[DATE_COLUMN]).dt.date
    act[DATE_COLUMN] = pd.to_datetime(act[DATE_COLUMN]).dt.date

    merged = preds.merge(act, on=[SERIES_COLUMN, DATE_COLUMN], how="inner")
    if merged.empty:
        msg = "no overlapping (series, date) keys between predictions and actuals"
        raise ValueError(msg)

    # Partial-overlap visibility: an inner join silently drops prediction rows that found no actual
    # (and actual rows with no prediction). Scoring then runs over a smaller support than either
    # side — loud enough to matter for cross-backend comparability, so log it with the counts.
    dropped_preds = len(preds) - len(merged)
    dropped_acts = len(act) - len(merged)
    if dropped_preds or dropped_acts:
        logger.warning(
            "partial overlap: scored over %d of %d prediction rows and %d actual rows "
            "(%d predictions and %d actuals had no (series, date) match)",
            len(merged),
            len(preds),
            len(act),
            dropped_preds,
            dropped_acts,
        )
    return merged


def _summary(merged: "pd.DataFrame", quantiles: list[float]) -> dict[str, float]:
    """Compute the aggregate metric scalars from an already-joined frame (with ``actual``)."""
    import numpy as np  # noqa: PLC0415 - lazy heavy import

    actual = merged[_ACTUAL_COLUMN].to_numpy(dtype=float)
    forecast = merged[FORECAST_COLUMN].to_numpy(dtype=float)
    error = forecast - actual

    # sMAPE: symmetric, guarded so a both-zero (perfect) point contributes 0 rather than NaN.
    # Mask the division itself (not just np.where) so no divide-by-zero is ever evaluated.
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    smape = np.zeros_like(denom)
    np.divide(np.abs(error), denom, out=smape, where=denom != 0.0)

    # Mean pinball loss across the requested quantiles (over all points x quantiles).
    pinball = []
    for q in quantiles:
        pred = merged[quantile_column(q)].to_numpy(dtype=float)
        delta = actual - pred
        pinball.append(np.maximum(q * delta, (q - 1.0) * delta))
    quantile_loss = float(np.mean(np.concatenate(pinball))) if pinball else 0.0

    # Demand-normalized errors: MAE / RMSE expressed relative to actual demand, so absolute quality
    # reads as a fraction of typical volume (robust to zero-fill days — an aggregate ratio, not a
    # per-row divide). Guarded like BigQuery's SAFE_DIVIDE: a zero denominator yields NaN, not inf.
    rmse = float(np.sqrt(np.mean(np.square(error))))
    actual_sum = float(np.sum(actual))
    actual_mean = float(np.mean(actual))
    pmae = float(np.sum(np.abs(error))) / actual_sum if actual_sum != 0.0 else float("nan")
    prmse = rmse / actual_mean if actual_mean != 0.0 else float("nan")

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "smape": float(np.mean(smape) * 100.0),
        "quantile_loss": quantile_loss,
        "pmae": pmae,
        "prmse": prmse,
        "n_points": float(len(merged)),
    }
