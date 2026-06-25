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

## Data caveats (verified 2026-06-25 against project `hybrid-vertex`)

- `bigquery-public-data.new_york_citibike.citibike_trips` is **frozen**: date span
  **2013-07-01 → 2018-05-31**, ~58.9M trips. The 2018-05-31 max date dictates split boundaries
  (TEST = last 14 days, etc.).
- `bigquery-public-data.noaa_gsod` LaGuardia station (`stn=725030`, `wban=14732`) has full daily
  coverage over the window (2,556 days across 2013–2019) — station ids confirmed correct.
- `bigquery-public-data.new_york.citibike_stations` is a **current snapshot**, joined by
  station *name*. **6 of the top-25** station names do **not** match → those rows get null
  `capacity`/`region_id`/lat/long. Treat capacity as an approximation.
  - **Possible refinement:** carry `start_station_id` through the source query and join metadata
    on `station_id` (more stable than names) to reduce the unmatched count, while keeping
    `start_station_name` as the series key. Not yet applied — flagged at the Stage 1 checkpoint.

## Tooling gotcha

- Pre-commit hooks defined in `.pre-commit-config.yaml` are **not active** until
  `uv run pre-commit install` is run. Defining ≠ installing.
