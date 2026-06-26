# geapTimes — Stage 3 Plan: Experiment Tracking & DOE Framework

> Archived snapshot of the approved Stage 3 plan, frozen at approval. Immutable — see
> [`PLANS.md`](../../PLANS.md) and git for the as-built record.

## Context

Stages 1–2 delivered the typed config contract (`geaptimes.schemas`), the BigQuery data layer
(`geaptimes.data.queries`, `geaptimes.naming`), and the **model layer**: a single `Forecaster`
ABC with `fit()` / `predict() -> ForecastResult`, the three backends (TimesFM, BQML, AutoML), and
`ForecastFactory.from_config(cfg) -> list[Forecaster]`. Every backend returns the same long
prediction frame (`PREDICTION_COLUMNS`). All of Stage 2 is **code-complete but never executed**.

Stage 3 builds the harness that **drives** those forecasters: a config-driven Design-of-Experiments
matrix, Vertex AI experiment tracking, GCS artifacts, and minimal point metrics — and it is where
TimesFM and BQML **execute live for the first time** on `hybrid-vertex`. The intended outcome: run
`run_experiment(cfg)` and get one tracked `ExperimentRun` per (DOE variant × enabled model), each
carrying logged params, MAE/RMSE, and prediction/config artifacts in
`gs://<bucket>/experiments/<name>/<run>/`.

## Locked decisions (from discussion)

- **DOE**: config-driven. A minimal `doe:` block in YAML — `axes` as dotted-path → list of values
  — cross-product-expanded into `ExperimentConfig` variants (re-validated through Pydantic). Empty
  axes ⇒ a single point (the base config). The run grid = variants × enabled models.
- **Live runs**: TimesFM + BQML execute live on `hybrid-vertex` this stage. AutoML is a
  **first-class backend in the runner** (no special-casing) but stays `enabled: false` in
  `base_config.yaml`; add `budget_milli_node_hours` to `AutoMLParams` so a future deliberate AutoML
  run is budget-controlled. No live AutoML run in Stage 3.
- **Metrics**: minimal MAE/RMSE on the point forecast vs TEST actuals now, so logged runs are
  meaningful. The full standardized suite (MAPE, quantile loss, per-series, comparison report)
  is Stage 5.
- **Layout**: a dedicated `src/geaptimes/experiment/` subpackage (deliberate refinement of the
  roadmap's "DOE in utils/"; mirrors the Stage 1 schema-location deviation — keeps the three
  concerns cohesive). Demo is `data_notebooks/03_experiment_tracking.ipynb`; the polished
  `scripts/run_experiment.py` CLI stays in Stage 5.
- **Testability**: identical to Stage 2 — the Vertex/GCS seam and the forecasters/actuals loaders
  are injected, so the entire harness unit-tests offline. The live run is a separate, explicit
  verification step (unlike Stage 2's "none run").

## Stage 3 — Detailed Items (Tier 2)

### 3.1 Schema + config: `doe:` block and AutoML budget
- `schemas.py`: add `DOEConfig(_Base)` with `axes: dict[str, list[Any]] = {}`; add
  `doe: DOEConfig = Field(default_factory=DOEConfig)` to `ExperimentConfig`. Add
  `budget_milli_node_hours: int = 1000` to `AutoMLParams` (validator: `> 0`).
- `config/base_config.yaml`: add `doe:\n  axes: {}` with a commented sweep example
  (`# forecast.horizon: [7, 14]`); add `budget_milli_node_hours: 1000` under the automl params.
- Tests: `doe` defaults to empty; axes round-trip; budget validator rejects ≤ 0.

### 3.2 `experiment/doe.py` — matrix engine (pure, offline)
- `@dataclass(frozen=True) DOEPoint(config: ExperimentConfig, overrides: dict[str, Any])`.
- `expand(cfg) -> list[DOEPoint]`: cross-product over `cfg.doe.axes`; for each combo, deep-copy via
  `cfg.model_dump()`, set each dotted path with a `_set_path` helper, drop the `doe` key, and
  re-validate with `ExperimentConfig.model_validate`. Stable ordering; empty axes ⇒ one point with
  `overrides={}`. Base `cfg` is never mutated.
- `point_slug(overrides) -> str`: short token for run names (e.g. `horizon14`), `"base"` if empty.
- Tests: grid size = product of axis lengths; overrides applied + re-validated; invalid override
  (e.g. negative horizon) raises via Pydantic; base config unchanged.

### 3.3 `experiment/metrics.py` — minimal point metrics (pure, offline)
- `point_metrics(predictions, actuals, *, series_col, time_col, target_col) -> dict[str, float]`:
  inner-join `predictions[series,date,forecast]` to `actuals[series,date,target]` on (series, date),
  return `{"mae":…, "rmse":…, "n_points":…}`. Raises if the join is empty (mismatched windows).
- Reuses `base.SERIES_COLUMN/DATE_COLUMN/FORECAST_COLUMN`.
- Tests: known frame → exact MAE/RMSE; partial-overlap join counts; empty-overlap raises.

### 3.4 `experiment/tracking.py` — Vertex Experiments + GCS artifacts (injected seam)
- `ExperimentTracker` wrapping `aiplatform`: `__init__(cfg, *, aiplatform=None, artifact_sink=None)`
  (lazy `from google.cloud import aiplatform`; default `artifact_sink` uploads bytes to GCS via
  `storage.Client`). `run(run_name)` is a context manager doing
  `aiplatform.init(project, location, experiment=cfg.execution.experiment_name, staging_bucket=…)`
  → `start_run(run_name)` → `yield` → `end_run()`.
- `log_params(dict)` / `log_metrics(dict)` delegate to `aiplatform.log_params/log_metrics`
  (coerce values to primitives). `log_artifacts(point, result, run_uri)` writes resolved config
  JSON + predictions CSV under `gs://<bucket>/experiments/<exp>/<run_uri>/` via the sink.
- All params/metrics/artifact writes recorded so a fake tracker (or fake aiplatform + list sink)
  captures everything offline. Tests: lifecycle order, params/metrics forwarded, artifact URIs +
  payloads written, no real cloud calls.

### 3.5 `experiment/runner.py` — orchestration
- `@dataclass RunRecord(run_name, model, overrides, metrics, metadata, artifact_uri)`.
- `run_experiment(cfg, *, tracker=None, forecasters=None, actuals_loader=None, now=None) -> list[RunRecord]`:
  `expand(cfg)` → for each point, `ForecastFactory.from_config(point.config)` (or injected
  `forecasters`), and for each forecaster open `tracker.run(run_name)`, call `fit()` → `predict()`,
  load TEST actuals (`actuals_loader(point.config)`; default reads prepped TEST rows from BQ via
  `table_names`), compute `point_metrics`, then `log_params` (model type, horizon, context, axis
  overrides, device/backend metadata) + `log_metrics` + `log_artifacts`. `run_name =
  f"{fc.name}-{point_slug(overrides)}-{timestamp}"` (timestamp from injected `now`,
  default `datetime.now(UTC)`).
- Default tracker = `ExperimentTracker(cfg)`. Tests inject a fake tracker, fake forecasters
  returning canned `ForecastResult`s, a fake actuals loader, and fixed `now` → assert grid size,
  per-run params/metrics/artifacts, and `RunRecord`s. Fully offline.

### 3.6 AutoML budget wiring
- `automl.py`: add `budget_milli_node_hours=self.params.budget_milli_node_hours` to `run_kwargs()`
  (first-class; unchanged dispatch). Test asserts it flows through `run_kwargs`/`fit`.

### 3.7 Demo — `data_notebooks/03_experiment_tracking.ipynb`
- Load `base_config.yaml` → `run_experiment(cfg)` (TimesFM + BQML enabled) → render the
  `RunRecord` table (model, MAE, RMSE, artifact URI) and a forecast-vs-TEST plot with the q10–q90
  band. `geaptimes` kernel pinned; Colab opt-in cell. Authored to run live in 3.8.

### 3.8 Live run + notes (hybrid-vertex)
- Pre-req: TEST-window actuals require the **prepped** table (already live:
  `citibike_daily_prepped__top25_h14_t14_v14`). BQML also needs the **train**/**infer** tables —
  build them via the Stage 2.2 builders (`build_train_query`/`build_infer_query`) before the BQML
  run (one-time, labelled DDL). TimesFM needs neither (reads prepped directly).
- Execute `run_experiment` live: TimesFM (downloads 200M once, forecasts), BQML (CREATE MODEL +
  ML.FORECAST). Confirm one Vertex `ExperimentRun` per model with params + MAE/RMSE, and artifacts
  under `gs://geaptimes-hybrid-vertex/experiments/citibike-daily-baseline/…`.
- Record outcomes (run names, metric values, artifact paths, any BQML/Vertex output-schema quirks)
  in `docs/notes/`.

## Files

- **New**: `src/geaptimes/experiment/__init__.py`, `doe.py`, `metrics.py`, `tracking.py`,
  `runner.py`; `data_notebooks/03_experiment_tracking.ipynb`; `tests/test_doe.py`,
  `test_metrics.py`, `test_tracking.py`, `test_runner.py`.
- **Edit**: `src/geaptimes/schemas.py` (DOEConfig + `doe`; AutoML budget); `config/base_config.yaml`
  (`doe:` block + budget); `src/geaptimes/models/automl.py` (budget in `run_kwargs`);
  `tests/test_automl.py` (budget assertion); `tests/` schema test for `doe`; `PLANS.md`
  (Tier 1 → Stage 3 active; Tier 2 items 3.1–3.8 with hashes); `pyproject.toml` only if a new
  per-file lint ignore is needed.
- **Dependency**: `uv add google-cloud-storage` (used for artifact upload; currently only transitive
  via aiplatform — declare it explicitly).
- **Reuse**: `geaptimes.schemas`, `geaptimes.naming.table_names`, `geaptimes.models.factory`,
  `geaptimes.models.base` (column constants + `QUANTILES`), `geaptimes.constants.RESOURCE_LABELS`,
  `geaptimes.utils.logger`, `geaptimes.data.queries` (train/infer builders for 3.8).

## Verification

- **Offline** (gating): `uv run ruff check .` → `uv run ruff format --check .` →
  `uv run ty check` → `uv run pytest`. Then
  `GCP_PROJECT=hybrid-vertex uv run python -c "from geaptimes.schemas import ExperimentConfig;
  from geaptimes.experiment.doe import expand;
  print(len(expand(ExperimentConfig.from_yaml('config/base_config.yaml'))))"` → DOE expands
  (1 with empty axes) with no cloud calls.
- **Live** (this stage, manual/auth): build train/infer tables, then run the notebook /
  `run_experiment` for TimesFM + BQML on `hybrid-vertex`; confirm `ExperimentRun`s in the Vertex
  console + GCS artifacts; record in `docs/notes/`.
- **STOP** checkpoint: present the DOE engine, tracking wrapper, runner, metrics, the AutoML
  budget wiring, and the live run results.

## Out of scope (later stages)

- KFP pipelines + TimesFM cloud-inference container + comparison DAG (Stage 4); standardized
  MAE/RMSE/MAPE/Quantile-Loss eval suite + comparison report + `scripts/run_experiment.py` CLI
  (Stage 5); live AutoML execution (wired now, run when explicitly budgeted); TimesFM XReg
  covariates (opt-in, deferred).
