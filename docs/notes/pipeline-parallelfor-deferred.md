# `dsl.ParallelFor` in the comparison pipeline — deferred (2026-06-26)

**Decision:** Do **not** use `dsl.ParallelFor` for the cross-service (three-backend) axis of the
comparison DAG. Revisit only for a *homogeneous inner axis* (see below). Holding for now.

## Why not for the three regimes

1. **Backend-level parallelism already exists, for free.** KFP runs tasks with no edges between
   them concurrently. The first live run confirmed it: `run-backend` (BQML), `run-backend-2`
   (AutoML), and `register-timesfm` were all `RUNNING` in the same poll sample. `ParallelFor` would
   not add concurrency we don't already have.
2. **`ParallelFor` wants homogeneous loop bodies; our backends are heterogeneous.** It runs the
   *same* sub-graph per item (cf. the reference DAG below, which iterates same-shape `pairs_json`
   items through one component chain). Our three backends have genuinely different chains:
   - BQML: `CREATE MODEL` → `ML.FORECAST` (two BQ queries)
   - AutoML: `TimeSeriesDataset.create` → `AutoMLForecastingTrainingJob.run` → `batch_predict` → flatten
   - TimesFM: **no training** → register → deploy → endpoint/batch predict
   To force these into one loop body you'd either special-case TimesFM out, or put a
   `switch(model_type)` *inside* the loop — re-introducing the mega-component we are explicitly
   removing (per-service single-responsibility components; see the Stage 4 redesign).
3. **Conflicts with a locked decision.** Branches in this pipeline resolve **at compile time from
   typed config** (not runtime `dsl.Condition`/`ParallelFor`). The DRY/scalability benefit people
   reach to `ParallelFor` for is instead obtained here via a **compile-time Python loop + a
   per-service chain-builder** (`add_backend_chain(cfg, model)`), which keeps heterogeneity explicit
   and preserves the compile-time-resolution decision.

## When `ParallelFor` *would* earn its keep

A genuinely **homogeneous** fan-out — the same component chain over N same-shape configs:
- one backend swept over many **DOE / hyperparameter variants**
- the same backend over multiple **backtest windows** or **series groups**

If/when we want that inner axis, `ParallelFor` is the right tool. The cross-service comparison is
not that axis.

## Reference

Prior usage pattern (same-shape items, one chain per item, sequential via `parallelism=1`):
`https://github.com/tottenjordan/gepa-prompt-wrangler/blob/main/wrangler/pipeline/dag.py` —
three `ParallelFor` blocks over `pairs_json`; the most complex chains
`optimize → redeploy → eval(after)` inside one loop body so KFP data dependencies (and caching)
flow correctly. That homogeneity is exactly what our cross-service axis lacks.
