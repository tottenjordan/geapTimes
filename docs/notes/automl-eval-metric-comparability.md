# AutoML eval-metric comparability — audit + hardening

Motivated by the observation that **AutoML eval metrics haven't looked comparable** to the other
backends in test runs. This note records the audit of the shared scoring path, the conclusion (no
scoring bug), the three comparability-integrity gaps it surfaced, and the hardening that closed them.

## The shared scoring path

Every backend funnels its predictions into **one** scorer — there is no per-backend metric code:

- BQML (`ARIMA_PLUS_XREG`) and AutoML train→infer **in-process**; TimesFM is **served** and predicted
  against an endpoint. All three emit the standardized prediction shape (`series`/`date`/`forecast` +
  `qXX` quantile columns).
- `score_and_track_step` (`src/geaptimes/pipelines/steps.py`) →
  `evaluate()` (`src/geaptimes/experiment/metrics.py`) computes MAE / RMSE / sMAPE / quantile-loss /
  `n_points` over the **inner join** of predictions to the held-out TEST actuals on `(series, date)`.
- `compare_step` (`steps.py`) → `rank_backends` (`src/geaptimes/experiment/comparison.py`) ranks by
  RMSE (tie-break MAE) and renders the Markdown table.

Because one `evaluate()` feeds both the offline runner and the pipeline scorer, "identical backtest"
is enforced by construction — the same join, the same metric formulas, for all backends.

## Conclusion: no scoring / alignment bug

Confirmed live (5.5 warm-reuse run, `--disable-automl`) that TimesFM and BQML both scored over
`n_points = 308`. Earlier successful AutoML runs also scored over `n_points = 308` — **identical
support** to the other backends. So the metric gap is **not** an alignment or denominator artefact:

- AutoML's MAE ≈ 142 / RMSE ≈ 183 vs TimesFM ≈ 86 / 114 and BQML ≈ 78 / 102 is **real
  underperformance at the floor training budget** (AutoML Forecasting was run at its minimum node-hour
  budget for cost), not a scoring defect.

### Resolved: AutoML's full metric suite confirmed live (2026-07-01)

sMAPE and quantile-loss showed as `NaN` for AutoML in older comparison tables only because **every
successful AutoML run predated the Stage 5.2 scorer** (which added sMAPE + quantile-loss). The 5.5
live run used `--disable-automl`, so AutoML had never been scored by the current suite live.

The full 3-backend confirmation run
(`geaptimes-comparison-top25-h14-t14-v14-20260701040623`, `--no-cache`, all 19 tasks SUCCEEDED)
closed this. All three backends scored over **`n_points = 308`** — exact parity — and AutoML's full
suite was populated live for the first time:

| backend | rmse | mae | sMAPE | quantile_loss | n_points |
| --- | --- | --- | --- | --- | --- |
| bqml_arima_xreg (winner) | 101.72 | 77.72 | 42.66 | 30.27 | 308 |
| timesfm | 113.65 | 85.51 | 39.26 | 31.30 | 308 |
| automl | 187.71 | 147.78 | 63.93 | 65.11 | 308 |

The AutoML gap holds across **every** metric (not just MAE), reconfirming real floor-budget
underperformance rather than a scoring artefact. Because `n_points` is identical across backends, the
new parity check correctly emits **no** warning — the happy-path behaviour of the WS-A hardening,
verified live. `teardown-serving` succeeded, so the TimesFM endpoint + uploaded served model were
torn down (`keep_deployed=false`); AutoML's registry model is a batch-predict training artefact and
is not endpoint-governed.

## Comparability-integrity gaps found (and closed)

Even without a bug, the scorer could *silently* compare non-comparable numbers. Three gaps, now fixed:

1. **No cross-backend `n_points` parity check.** `_merge` only raised on an *empty* overlap; a backend
   with *partial* overlap scored over fewer points and `rank_backends` then compared metrics over
   **different denominators** with nothing flagging it.
   → `rank_backends` now records a `Comparison.warnings` entry when backends have differing non-zero
   `n_points` ("backends scored over differing point counts …"). `render_ranking_markdown` prints the
   warnings under the table; `compare_step` logs them at WARNING.

2. **`n_points` was invisible in the comparison.** The Markdown table only showed the error metrics, so
   a reviewer couldn't see whether backends shared a denominator.
   → `n_points` is now a **display-only** column in the ranking table (integer-formatted). It is
   **never** a sort key — ranking stays RMSE→MAE.

3. **Ranking was not NaN-safe.** `metrics.get(m, inf)` let a present-but-`NaN` metric (the pre-5.2
   shape) pass through as NaN, making `.sort()` order undefined.
   → the sort key coerces any non-finite value (NaN/±inf/missing) to `+inf`, so a degenerate metric
   sinks the model rather than corrupting the order.

Additionally, `_merge` now **logs a WARNING on partial overlap** (counts of prediction/actual rows
that found no `(series, date)` match). The empty-overlap `ValueError` is unchanged, and metric values
themselves are unchanged — this is pure visibility.

## Follow-on: demand-normalized metrics + payload completeness (2026-07-01)

Comparing our scorer to statmike's `Vertex AI AutoML Forecasting - Python client` SQL (its "Review
Custom Metrics with SQL" cells) confirmed **MAE and RMSE are formula-identical** (`AVG(ABS(diff))`,
`SQRT(AVG(POW(diff,2)))`). Two deliberate divergences remain: he uses classic MAPE
(`AVG(SAFE_DIVIDE(ABS(diff), actual))`, which silently drops zero-actual days via `NULL` — the
denominator trap we avoid), and he has no pinball/quantile loss (we do). We adopted the one thing his
SQL had that we lacked — **demand-normalized error**:

- `pmae = SUM(|error|) / SUM(actual)` and `prmse = RMSE / AVG(actual)` — MAE/RMSE as a fraction of
  typical demand, so absolute quality reads scale-free. Aggregate ratios (not per-row divides), so
  robust to zero-fill; guarded like `SAFE_DIVIDE` (zero denominator → `NaN`). Display-only, **never**
  sort keys; ranking stays RMSE→MAE.

Wiring these in surfaced a **latent gap**: the pipeline's `score_and_track` component only put the
four ranking metrics into the base64 compare payload, so `compare_backends` → `rank_backends`
received **no `n_points`** — the WS-A `n_points` column and parity check silently rendered `NaN`
**in the live pipeline path** (they only ever worked in the offline `run_experiment` CLI, which
passes full metric dicts). Fixed by adding `pmae`/`prmse`/`n_points` to the compare payload
(`src/geaptimes/pipelines/components.py`); task-UI `system.Metrics` stays the four finite ranking
metrics (avoids `NaN` in the artifact). The `n_points` parity check is now actually functional in the
pipeline. **Requires a runtime-image rebuild before the next live run** (components run baked code).

**Confirmed live (2026-07-01, `...20260701111422`, image `sha256:1c110c58…`, all 19 tasks
SUCCEEDED):** the rendered `ranking.md` artifact now carries the `pmae`/`prmse`/`n_points` columns,
all three backends at `n_points=308` (parity → no warning), and the ExperimentRuns log `pmae`/`prmse`
alongside the ranking metrics. Live table (winner bqml_arima_xreg):

| rank | model | mae | rmse | smape | quantile_loss | pmae | prmse | n_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | bqml_arima_xreg | 77.72 | 101.72 | 42.66 | 30.27 | 0.2733 | 0.3576 | 308 |
| 2 | timesfm | 85.51 | 113.65 | 39.26 | 31.30 | 0.3007 | 0.3996 | 308 |
| 3 | automl | 144.62 | 184.41 | 62.18 | 62.48 | 0.5085 | 0.6484 | 308 |

pMAE reads directly: BQML's error is ~27% of average daily demand, TimesFM ~30%, AutoML ~51% — the
same ranking as RMSE, now scale-free. (AutoML's numbers shift run-to-run — 144.6/184.4 here vs
147.8/187.7 on the prior run — expected non-determinism at the floor budget; still last on every
metric.)

## AutoML point-metric gap: median-underprediction bias (2026-07-01)

Follow-up to the recurring "AutoML metrics look off" concern. Investigation (a) — pulling the live
batch-prediction output (`predictions_...T06_29_55...`, this run's AutoML) and joining to actuals —
isolated the cause. **The metrics are computed identically to the other backends; the gap is the
objective, not the math.**

Under `minimize-quantile-loss`, AutoML's point forecast `.value` **is the median (q50)**. Daily
per-station trip counts are right-skewed, so the median sits well below the mean:

| quantity | value |
| --- | --- |
| mean actual | 284.4 |
| mean forecast (= median q50) | 168.1 (~59% of demand) |
| mean signed error (bias) | **−116.3** (systematic underprediction) |
| fraction of points under-predicted | **82%** |
| MAE / RMSE (as reported) | 144.6 / 184.4 |
| RMSE with global bias removed (variance only) | 143.1 |
| MAE using q90 as the point instead | 113.4 (≈ TimesFM) |

So ~40% of AutoML's MSE is pure bias, and simply choosing a higher quantile as the point estimate
pulls MAE into TimesFM territory with the *same model*. Even fully debiased (~143 RMSE) AutoML still
trails BQML/TimesFM, so a residual floor-budget gap remains — but the headline 2× gap is the
point-estimate/objective choice.

**Fix applied:** switch AutoML to `optimization_objective: minimize-rmse` (targets the conditional
*mean*, eliminating the median bias — this is why the statmike notebook uses `minimize-rmse`).
Trade-off: Vertex allows quantile outputs only with `minimize-quantile-loss`, so the run becomes
**point-only** — AutoML emits no quantiles and its `quantile_loss` is reported as `NaN` (honest; the
scorer does not fabricate intervals). Point metrics (MAE/RMSE/sMAPE/MAPE/pMAE/pRMSE) become an
apples-to-apples comparison against the other backends' mean-targeting point forecasts. The residual
budget lever remains available separately.

## Point-only NaN-sink bug + minimize-rmse live confirmation (2026-07-01)

Applying the `minimize-rmse` switch surfaced a **latent NaN-handling bug** in the metric sinks. The
first live run under the new objective (`…20260701150736`, `--no-cache`) trained and inferred AutoML
fine (`train-backend-2` + `infer-backend-2` SUCCEEDED) but **failed at `score-and-track-2`**:

```
InvalidArgument: 400 Value for the field 'quantile_loss' is invalid.
  Numeric values can not be NaN or Infinity.
```

Point-only AutoML reports `quantile_loss = NaN` (no quantiles produced — honest), and
`ExperimentTracker.log_metrics` was shipping the full metric dict, NaN included, to Vertex's
**ExperimentRun metadata service**, which rejects non-finite numbers outright. TimesFM/BQML passed
because their `quantile_loss` is finite; only the point-only backend tripped it.

**Fix (`b476abc`):** guard both metric sinks that reach a NaN-hostile layer, keep the value everywhere
else:
- `ExperimentTracker.log_metrics` (`src/geaptimes/experiment/tracking.py`) drops non-finite entries
  before the Vertex call and logs a WARNING naming what it dropped.
- the KFP `system.Metrics` UI artifact (`score_and_track` in `src/geaptimes/pipelines/components.py`)
  logs only the **finite** ranking metrics (`math.isfinite` guard).
- the base64 **compare payload is unchanged** — it still carries `quantile_loss=NaN`, which
  round-trips through Python `json` and renders in the ranking table; `rank_backends` is already
  NaN-safe, so the NaN sinks the metric without corrupting the sort. We persist honesty everywhere
  except the two sinks that physically can't store NaN.

**Confirmed live** on the rebuilt image (`sha256:0569f19f…`): run `…20260701183919`, **all 19/19
tasks SUCCEEDED** (including `score-and-track-2`), `n_points=308` parity (no warning). The
`minimize-rmse` objective removed the median bias as predicted:

| backend | mae | rmse | smape | quantile_loss | mape | pmae | prmse | n_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bqml_arima_xreg (winner) | 77.72 | 101.72 | 42.66 | 30.27 | 0.434 | 0.273 | 0.358 | 308 |
| timesfm | 85.51 | 113.65 | 39.26 | 31.30 | 0.457 | 0.301 | 0.400 | 308 |
| automl | 130.33 | 169.51 | 55.09 | **NaN** | 0.420 | 0.458 | 0.596 | 308 |

AutoML's point metrics dropped vs the biased `minimize-quantile-loss` run: **MAE 144.6 → 130.3
(−10%), RMSE 184.4 → 169.5 (−8%)**. It still ranks last on RMSE/MAE — the residual floor-budget gap
(even fully debiased, RMSE was ~143), not the median bias, which is now gone. Note AutoML's **MAPE is
actually best** (0.420): classic MAPE weights low-demand days (where the mean-targeting model now does
well), while pMAE/pRMSE weight high-demand stations (where it lags) — exactly why MAPE is display-only
and never a ranking key.

## Takeaways

- The comparison is structurally fair: one shared scorer, one inner join, identical formulas.
- The AutoML gap is genuine (floor budget), not a scoring artefact — proven by `n_points` parity.
- Non-comparability is now **loud**: differing point counts warn, `n_points` is on the table, and the
  ranking can't be corrupted by a NaN metric.
- Closed: the 2026-07-01 3-backend run produced AutoML's full sMAPE/quantile-loss suite live under the
  current scorer with `n_points = 308` parity — no open follow-ups remain.
