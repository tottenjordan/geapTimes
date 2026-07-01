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

### Still open: AutoML's full metric suite has never been produced live

sMAPE and quantile-loss showed as `NaN` for AutoML in older comparison tables only because **every
successful AutoML run predates the Stage 5.2 scorer** (which added sMAPE + quantile-loss). The 5.5
live run used `--disable-automl`, so AutoML has **never** been scored by the current suite live. A
full 3-backend confirmation run (~2.5 h + AutoML budget) would close this; it is offered separately
and is **not** required by the hardening below.

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

## Takeaways

- The comparison is structurally fair: one shared scorer, one inner join, identical formulas.
- The AutoML gap is genuine (floor budget), not a scoring artefact — proven by `n_points` parity.
- Non-comparability is now **loud**: differing point counts warn, `n_points` is on the table, and the
  ranking can't be corrupted by a NaN metric.
- Open follow-up: run the full 3-backend suite once to record AutoML's sMAPE/quantile-loss under the
  current scorer (all prior AutoML runs predate it).
