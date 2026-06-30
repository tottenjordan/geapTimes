# Stage 5 — Standardized Evaluation & Entry Point (approved 2026-06-30)

Immutable snapshot of the approved Stage 5 plan. Living status lives in
[`PLANS.md`](../../PLANS.md); course corrections are tracked there + in git, not by editing this file.

## Goal

Deliver the standardized evaluation suite and the polished entry point promised by the Tier 1
roadmap: *uniform eval (MAE / RMSE / MAPE / Quantile Loss) over identical backtests; finalize
`scripts/run_experiment.py`.* Plus one deferred item folded in by user decision: **warm-endpoint
reuse** (the optional-quantiles switch stays in the backlog).

## Context (what exists today)

- `experiment/metrics.py` → only `point_metrics` (MAE / RMSE).
- `experiment/runner.py` → `run_experiment()` works but is library-only; there is **no**
  `scripts/run_experiment.py` (only `demo_timesfm.py` + `setup_gcp.py`).
- The pipeline scores via `score_and_track_step` + `compare_backends` — MAE / RMSE only.
- All backends always emit `qXX` quantile columns (optional-quantiles not adopted), so quantile
  loss is always computable.
- Warm-endpoint reuse design is captured in `docs/notes/timesfm-endpoint-reuse-design.md`.

## Scope decision (2026-06-30)

- **Core (fixed):** eval suite + unify into the pipeline scorer + `run_experiment.py` CLI.
- **Folded in:** warm-endpoint reuse (needs one live re-run to validate).
- **Backlog (not now):** optional-quantiles config switch (CODE_STANDARDS deferred section).

## Items

### 5.1 — Standardized eval suite (`experiment/metrics.py`)

Add, alongside the retained `point_metrics`:

- **Guarded sMAPE** — symmetric MAPE with a small-denominator guard. Raw MAPE divides by ~0 on the
  zero-filled low-count stations, so sMAPE (and a documented guard) is the safe choice.
- **Pinball / quantile loss** — averaged across `QUANTILES` (`[0.1, 0.3, 0.5, 0.7, 0.9]`) using the
  standardized `qXX` columns.
- **Per-series breakdown** — the same metrics grouped by `series` (returns a long/dict structure).
- **`evaluate()` aggregator** — one entry point returning the standardized summary
  (`mae`, `rmse`, `smape`, `quantile_loss`, `n_points`, plus the per-series breakdown), back-compatible
  superset of `point_metrics`.

Offline unit tests for each (known inputs → known values, zero-denominator guard, empty-overlap raise).

### 5.2 — Unify the suite across both backtest paths

- Wire `evaluate()` into the offline `runner.py` and the pipeline `score_and_track_step` so both log
  the full suite from **one** implementation ("identical backtests").
- Factor a shared `rank_backends` helper (single ranking + Markdown rendering) consumed by both
  `compare_backends` (pipeline) and the CLI report (5.3). `compare_backends` ranks + renders on the
  richer metric set.
- Update compile / contract / step tests. The base64 fan-in hop and ranking key stay test-guarded.

### 5.3 — `scripts/run_experiment.py` CLI

- argparse wrapper over `run_experiment`: `--config`, `--enable-automl` / `--disable-automl`,
  `--dry-run`; prints a `RunRecord` table and writes a comparison report (reusing the 5.2 ranker).
- Injected seams (runner / tracker / clock) so the CLI logic unit-tests offline.

### 5.4 — Warm endpoints + fingerprint reuse + teardown guard

Per `docs/notes/timesfm-endpoint-reuse-design.md`:

- Support a **warm** mode (`pipeline.serving.keep_deployed = True`) where the endpoint is left
  running.
- `find_reusable_endpoint_step(cfg)` — lookup by label `fingerprint = sha256(image **digest** +
  serving env)` (+ `solution=geaptimes`); returns a healthy match's resource name or `""`. Resolve the
  AR **digest** behind `:latest` before hashing (never fingerprint the mutable tag).
- DAG branch: on a hit, skip register + deploy and feed the found endpoint into `endpoint-predict`;
  otherwise the current register → deploy → predict path.
- **Teardown guard:** the ExitHandler must only tear down what *this run created* — never a reused or
  shared warm endpoint. Track an owned/created-this-run flag.
- Offline compile + step tests (lookup seam injected). Reviews-as-gates on this item (the
  teardown-guard + concurrency are the riskiest surface).

### 5.5 — Verification + STOP

- Offline gate: `uv run ruff check .` → `uv run ruff format --check .` → `uv run ty check` →
  `uv run pytest`.
- **One cheap `--disable-automl` live run** validating warm deploy → reuse-on-second-run →
  guarded-teardown (shared endpoint survives), *and* (same run) the unified richer metric suite +
  Markdown live. Warm endpoints only touch TimesFM serving, so AutoML's ~2.5h is unnecessary; its
  richer-metrics path is covered by unit tests + the already-proven train/infer split.
- Record in `docs/notes/`; finalize PLANS; **STOP** for approval.

## Key files

- **Edit:** `src/geaptimes/experiment/metrics.py` (suite), `experiment/runner.py` (wire),
  `pipelines/steps.py` (`score_and_track_step`, `teardown_serving_step`, new
  `find_reusable_endpoint_step`), `pipelines/components.py` + `pipeline.py` (rank helper, reuse
  branch, teardown guard), `pipelines/config.py` (fingerprint), `schemas.py` (warm-mode knobs if
  needed), `scripts/run_experiment.py` (new).
- **Tests:** `tests/test_metrics.py` (new/extended), `test_pipeline_steps.py`,
  `test_pipeline_compile.py`, `test_pipeline_components.py`, `tests/test_run_experiment_cli.py` (new).
- **Docs:** `docs/notes/stage-5-*.md` (live run), update `timesfm-endpoint-reuse-design.md` status.

## Risks

- **Teardown guard** must be airtight — deleting a shared warm endpoint is a destructive regression.
  Covered by the reviews-as-gate + the live reuse run.
- **Fingerprint on digest, not `:latest`** — resolve the AR digest before hashing or risk serving a
  stale model.
- **sMAPE guard** — document the small-denominator behavior so the metric is interpretable.
- One cheap live run required (warm-endpoint path cannot be fully validated offline).
