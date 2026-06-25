# Forecasting: covariate taxonomy & per-model constraints

Captured 2026-06-25 during Stage 1 design. These are non-obvious constraints that shape the
data layer and config. Re-verify versions/columns before acting on them later.

## Covariate roles (official AutoML terminology)

We use the Vertex AI / GEAP AutoML feature-type names everywhere we can:

| Role | Meaning | geapTimes columns |
|------|---------|-------------------|
| `time_series_attribute_columns` | static, do not vary with time | `capacity`, `region_id`, `latitude`, `longitude` |
| `available_at_forecast_columns` | known at forecast request time (doc example: *predicted* weather) | `temp`, `prcp`, `day_of_week`, `month`, `is_weekend` |
| `unavailable_at_forecast_columns` | unknown before the forecast (doc example: *actual* weather) | `avg_tripduration`, `pct_subscriber`, `ratio_gender` |

## Why weather is "available at forecast"

It is a *mechanical* simplification, not a physical truth:
- **BQML `ARIMA_PLUS_XREG`** requires future regressor values to forecast at all → regressors
  must be available-at-forecast.
- In **backtests** we feed weather **actuals** over the horizon (proxy for a perfect forecast).
  This is mildly optimistic (assumes perfect weather foresight); documented as a caveat.
- The engineered trip-derived features (`avg_tripduration`, `pct_subscriber`, `ratio_gender`)
  are genuinely **unavailable at forecast** — you don't know tomorrow's avg trip duration.

## Per-model covariate handling

- **AutoML Forecasting:** handles all three roles via column declarations.
- **BQML `ARIMA_PLUS_XREG`:** can only use the **available-at-forecast** subset (XREG needs
  future values). The unavailable-at-forecast engineered features are excluded for BQML.
- **TimesFM 2.5:** **univariate** in v1 (per-series target arrays only). 2.5 re-added covariate
  support via XReg (`timesfm[xreg]`), but the reference notebook runs univariate; XReg is an
  opt-in later enhancement. → TimesFM consumes a univariate extract, effectively a 3rd dataset
  shape alongside AutoML (full table) and BQML (target + available-at-forecast).

## Holidays — use built-in support, don't hand-engineer

Both services model holidays natively (only valid at **daily** granularity, which we are):
- **AutoML:** `holiday_regions` parameter.
- **BQML:** `HOLIDAY_REGION` option in `CREATE MODEL`.
So holidays are a `holiday_region` config value, **not** a manual column. We still engineer
`day_of_week` / `month` / `is_weekend` manually. (Exact region code, e.g. `US`, to be confirmed
when wiring defaults — the docs page pulled didn't enumerate the codes.)

## TimesFM 2.5 — version & API facts

- `timesfm` **package** `==2.0.1` (PyPI/git tag `v2.0.1`) ships the **2.5 model** classes
  (`src/timesfm/timesfm_2p5/`). Package version ≠ model version. No git install needed.
- API: `timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")` →
  `model.compile(timesfm.ForecastConfig(max_context=..., max_horizon=..., ...))` →
  `point_forecast, quantile_forecast = model.forecast(horizon=H, inputs=<list of 1D arrays>)`.
- 2.5 supports up to 16k context (was 2048) and continuous quantile forecast. The
  multiple-of-32 context rule was a 2.0 patch-size constraint — re-verify for 2.5 before relying
  on it; we keep `context_len` a multiple of 32 (default 512) to be safe.

## Data caveats

- The BQ public table `bigquery-public-data.new_york_citibike.citibike_trips` is **frozen**
  (~2018). Verify the exact max date in the prep notebook; it dictates split dates.
- `bigquery-public-data.new_york.citibike_stations` is a **current snapshot** — its `capacity`
  is today's value, joined by station name to 2018 trips, so a few names may not match. Report
  unmatched stations in the notebook; treat as an approximation.

## Tooling gotcha

- Pre-commit hooks defined in `.pre-commit-config.yaml` are **not active** until
  `uv run pre-commit install` is run. Defining ≠ installing.
