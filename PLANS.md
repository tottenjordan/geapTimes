# geapTimes — Plans & Progress

Living tracker. **Tier 1** is a coarse master roadmap of all stages; **Tier 2** decomposes the
*active* stage into numbered items with status and commit hash. Later stages are expanded
just-in-time when reached. Item IDs are stable and map to commits.

Each approved per-stage plan is archived immutably under [`docs/plans/`](docs/plans/) at approval
time; this file is the living view of progress.

Status legend: `pending` · `in progress` · `done` · `blocked`

---

## Tier 1 — Master Roadmap

| Stage | Title | Status |
|-------|-------|--------|
| 1 | Data Architecture & Config Schema | done |
| 2 | Model Factory & Forecaster Abstractions | done |
| 3 | Experiment Tracking & DOE Framework | items done — STOP (awaiting approval) |
| 4 | Managed Pipelines & Cloud Orchestration | pending |
| 5 | Standardized Evaluation & Entry Point | pending |

**Stage 2** — `src/geaptimes/models/`: `base.py` (ABC), `factory.py`, `automl.py`,
`timesfm.py` (CPU/GPU), `bqml.py`; local TimesFM inference demo.
**Stage 3** — DOE matrix engine in `utils/`; wire `aiplatform.Experiment`/`ExperimentRun`;
per-run GCS path `gs://<bucket>/experiments/<name>/<timestamp>/`; log params + metrics.
**Stage 4** — KFP pipelines in `src/geaptimes/pipelines/`; TimesFM cloud-inference custom
container (Model Garden → endpoint); side-by-side comparison DAG.
**Stage 5** — uniform eval (MAE/RMSE/MAPE/Quantile Loss) over identical backtests; finalize
`scripts/run_experiment.py` CLI.

---

## Tier 2 — Stage 3 (Experiment Tracking & DOE Framework) — ACTIVE

A config-driven **Design-of-Experiments** matrix expands one base config into a grid of variants;
the **runner** drives the Stage 2 forecasters (their first live execution) and logs each run to
Vertex AI **Experiments** with params, minimal point metrics (MAE/RMSE vs TEST actuals), and
prediction/config **artifacts** in `gs://<bucket>/experiments/<name>/<run>/`. The Vertex/GCS seam
and forecaster/actuals loaders are injected, so the whole harness unit-tests offline; the live run
is a separate, explicit step. TimesFM + BQML run live on `hybrid-vertex`; AutoML is a first-class
backend in the runner but stays `enabled: false` (run when explicitly budgeted). Full standardized
evaluation (MAPE, quantile loss, comparison report) is deferred to Stage 5.

| # | Item | Status | Commit |
|---|------|--------|--------|
| 3.1 | `schemas.py` + `base_config.yaml` — `doe:` block + `AutoMLParams.budget_milli_node_hours` | done | `e070977` |
| 3.2 | `experiment/doe.py` — DOE matrix engine (`expand`, `DOEPoint`, `point_slug`) | done | `e57eabf` |
| 3.3 | `experiment/metrics.py` — minimal point metrics (MAE/RMSE) | done | `e093194` |
| 3.4 | `experiment/tracking.py` — Vertex Experiments + GCS artifacts (injected seam) | done | `4b86489` |
| 3.5 | `experiment/runner.py` — `run_experiment` orchestration + `RunRecord` | done | `e0ba7dc` |
| 3.6 | `models/automl.py` — wire `budget_milli_node_hours` into `run_kwargs` | done | `14b22d4` |
| 3.x | `models/bqml.py` — fix: label model via BigQuery API (CREATE MODEL rejects label OPTIONS) | done | `6f3eef7` |
| 3.7 | Demo — `03_experiment_tracking.ipynb` | done | `42f8078` |
| 3.8 | Live run on `hybrid-vertex` (TimesFM + BQML) + notes | done | `8d29e19` |

### 3.1 Schema + config
`DOEConfig(axes: dict[str, list])` + `ExperimentConfig.doe`; `AutoMLParams.budget_milli_node_hours`
(default 1000, validator `> 0`). `base_config.yaml` gets an empty `doe.axes` (commented sweep
example) and the AutoML budget.

### 3.2 `experiment/doe.py`
`expand(cfg)` cross-products `cfg.doe.axes` into re-validated `ExperimentConfig` variants
(deep-copy via `model_dump` + dotted-path override + `model_validate`); empty axes ⇒ one `base`
point; base cfg never mutated. `point_slug(overrides)` names runs.

### 3.3 `experiment/metrics.py`
`point_metrics(predictions, actuals, …)` inner-joins on (series, date) → `{mae, rmse, n_points}`;
raises on empty overlap. Full eval suite is Stage 5.

### 3.4 `experiment/tracking.py`
`ExperimentTracker` wraps `aiplatform` (`init`/`start_run`/`log_params`/`log_metrics`/`end_run`)
with a `run(run_name)` context manager; artifacts (resolved config JSON + predictions CSV) written
to `gs://<bucket>/experiments/<exp>/<run>/` via an injected sink (default GCS `storage.Client`).
Fully offline-testable with a fake aiplatform + list sink.

### 3.5 `experiment/runner.py`
`run_experiment(cfg, *, tracker, forecasters, actuals_loader, now)` → for each `DOEPoint` ×
enabled forecaster: `fit()` → `predict()` → metrics vs TEST actuals → log params/metrics/artifacts;
returns `RunRecord`s. All cloud seams injected for offline tests; default actuals loader reads
prepped TEST rows.

### 3.6 AutoML budget
`run_kwargs()` gains `budget_milli_node_hours` from `AutoMLParams`; backend remains default-off.

### 3.7 Demo
`data_notebooks/03_experiment_tracking.ipynb`: `run_experiment` → `RunRecord` table + forecast-vs-
TEST plot with q10–q90 band; `geaptimes` kernel pinned; first executed in 3.8.

### 3.8 Live run + notes
Built train/infer tables (Stage 2.2 builders); ran TimesFM + BQML live on `hybrid-vertex` under
experiment `citibike-daily-baseline`. Backtest over the 14-day TEST window (25 series, 308 points):
**TimesFM** MAE 85.51 / RMSE 113.65; **BQML ARIMA_PLUS_XREG** MAE 77.72 / RMSE 101.72. One Vertex
`ExperimentRun` per model with params + metrics; artifacts (config.json + predictions.csv) under
`gs://geaptimes-hybrid-vertex/experiments/citibike-daily-baseline/<run>/`; BQML model labelled
`solution=geaptimes`. Two findings recorded in `docs/notes/stage-3-experiment-tracking-live-run.md`:
(a) **Stage 2 bug fixed** — BQML `CREATE MODEL` can't carry resource labels in OPTIONS, so the model
is labelled via the BigQuery API post-create (`6f3eef7`); (b) a workstation pyOpenSSL/urllib3 quirk
breaks the Python GCS upload, mitigated in the run harness with a `gcloud storage cp` artifact sink
(package default sink unchanged).

### Stage 3 verification
- **Offline** (gating): `uv run ruff check .` → `uv run ruff format --check .` →
  `uv run ty check` → `uv run pytest`; then `expand(base_config)` lists the grid with no cloud calls.
- **Live** (manual, auth): build train/infer, run TimesFM + BQML, confirm `ExperimentRun`s +
  artifacts, record in `docs/notes/`.
- **STOP** checkpoint: present the DOE engine, tracking wrapper, runner, metrics, AutoML budget
  wiring, and the live run results.

---

## Tier 2 — Stage 2 (Model Factory & Forecaster Abstractions) — COMPLETE

All three backends are **code-complete but not executed** in Stage 2 (no model downloads, no BQ
jobs, no Vertex spend); first live runs happen in Stage 3. Forecasters lazy-load / inject clients
so they import and unit-test fully offline. Every backend returns the same `ForecastResult` shape
(`QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]` → columns `q10, q30, q50, q70, q90`).

| # | Item | Status | Commit |
|---|------|--------|--------|
| 2.1 | `models/base.py` — `Forecaster` ABC + `ForecastResult` + `QUANTILES` | done | `b8961e5` |
| 2.2 | `data/queries.py` — `train`/`infer` builders (closes deferred 1.10) | done | `6b7d30a` |
| 2.3 | `models/timesfm.py` — `TimesFMForecaster` (in-process, device-agnostic) | done | `475bf02` |
| 2.4 | `models/bqml.py` — `BQMLForecaster` (ARIMA_PLUS_XREG SQL) | done | `97a07f6` |
| 2.5 | `models/automl.py` — `AutoMLForecaster` (managed job config) | done | `052f7ba` |
| 2.6 | `models/factory.py` — `ForecastFactory` | done | `23c3ee8` |
| 2.7 | Demo — `02_timesfm_local.ipynb` + `scripts/demo_timesfm.py` | done | `0a4c4c2` |

### 2.1 `models/base.py`
`Forecaster` ABC (`fit()` / `predict() -> ForecastResult`), `ForecastResult` dataclass
(`predictions` long DataFrame + `metadata`), `QUANTILES` and canonical prediction column names as
module constants. No backend logic.

### 2.2 `data/queries.py` train/infer builders
`build_train_query` (rows where `splits != 'TEST'`) and `build_infer_query` (future `horizon`
rows per series; calendar features derivable from generated dates, weather/target NULL beyond the
frozen window — documented inline, refined in Stage 4). Both emit labelled `CREATE OR REPLACE
TABLE`. Closes the Stage 1.10 deferral.

### 2.3 `models/timesfm.py`
`TimesFMForecaster`: lazy `from_pretrained` + `compile(ForecastConfig(...))`; device from
`execution.target` with `cuda.is_available()` fallback; `fit()` no-op (zero-shot); `predict()`
pulls per-series context from the prepped table and maps point + deciles → standardized columns.
Model loader + dataframe loader injected for offline tests.

### 2.4 `models/bqml.py`
`BQMLForecaster`: `build_create_model_sql` (`ARIMA_PLUS_XREG`/`ARIMA_PLUS`, id/timestamp/data
cols, `holiday_region`, labels; XREG selects `available_at_forecast_columns`) +
`build_forecast_sql` (`ML.FORECAST`, XREG joins the `infer` table). BQ client injected; tests are
SQL-string + result-mapping only.

### 2.5 `models/automl.py`
`AutoMLForecaster`: builds `AutoMLForecastingTrainingJob` constructor + `.run()` kwargs from
config — column roles from `CovariateRoles`, `forecast_horizon`, `context_window`,
`data_granularity_unit='day'`, `predefined_split_column_name='splits'`, `quantiles=QUANTILES`,
`holiday_regions`, `labels`. Gated (not executed); `aiplatform` injected.

### 2.6 `models/factory.py`
`ForecastFactory.create(model_cfg, cfg)` dispatch on `params.type`; `from_config(cfg)` builds all
**enabled** models; unknown type raises.

### 2.7 Demo
`data_notebooks/02_timesfm_local.ipynb` (+ thin `scripts/demo_timesfm.py`): load config →
`ForecastFactory.create` TimesFM → `predict()` → plot vs TEST actuals with quantile band.
`geaptimes` kernel pinned. Authored + runnable; first executed in Stage 3.

### Stage 2 verification
- **Offline only** (per "none run"): `uv run ruff check .` → `uv run ruff format --check .` →
  `uv run ty check` → `uv run pytest`; then construct all enabled forecasters via
  `ForecastFactory.from_config(...)` without triggering any download/BQ/Vertex (lazy load).
- **STOP** checkpoint: present the abstraction, factory dispatch, the three backend wrappers, and
  the train/infer builders.

---

## Tier 2 — Stage 1 (Data Architecture & Config Schema) — COMPLETE

| # | Item | Status | Commit |
|---|------|--------|--------|
| 1.1 | Repo scaffold + standards + tracker | done | `1b9dc1d` |
| 1.2 | `src/geaptimes/schemas.py` — Pydantic v2 typed config | done | `c9d201a` |
| 1.3 | `config/base_config.yaml` — DOE config | done | `ac039fe` |
| 1.4 | `src/geaptimes/data/queries.py` + `utils/logger.py` | done | `a25a81d` |
| 1.5 | `data_notebooks/01_citibike_prep.ipynb` | done | `bc43525` |
| 1.6 | `scripts/setup_gcp.py` (+ `geaptimes.gcp`) | done | `416b4b2` |
| 1.7 | Notebook kernel standardization + cloud validation | done | `78908a4` |
| 1.8 | Data-layer hardening + end-to-end build on hybrid-vertex | done | `330893c` |
| 1.9 | GCP resource labels (`solution=geaptimes`) + README badges | done | `8b4d4e8` |
| 1.10 | Table naming convention (`geaptimes.naming`) | done | `9b36f70` |

### 1.1 Repo scaffold + standards + tracker
`git init`; `uv init --package` (name `geaptimes`, py3.11); `pyproject.toml` with ruff/pytest/ty
config + dependency groups; runtime deps via `uv add` (`aiplatform==1.158.0`, bigquery,
db-dtypes, pandas, pydantic>=2, pyyaml, `timesfm[torch]==2.0.1`); `.pre-commit-config.yaml`;
`CODE_STANDARDS.md`, `CLAUDE.md` (links to standards), `README.md`, `.gitignore` (commit
`uv.lock`), generated `requirements.txt`, `docs/notes/`; this `PLANS.md`.

### 1.2 `src/geaptimes/schemas.py` (Pydantic v2)
Typed models for project/data/forecast/execution + per-model params as a **discriminated union**
on `params.type` (TimesFM/BQML/AutoML). Validators (`context_len` multiple of 32, `horizon>0`,
`top_n>0`); `ExperimentConfig.from_yaml` with `${ENV}` substitution. Tests in `tests/`.
**Deviation:** schemas live in the package (`src/geaptimes/schemas.py`, importable as
`geaptimes.schemas`) instead of the blueprint's non-importable `config/schemas.py`; the YAML
data stays in `config/`.

### 1.3 `config/base_config.yaml`
Agreed DOE config: data filters/splits/weather/covariate roles/holiday_region; forecast
(`frequency:"D"`, `horizon:14`); models (timesfm univariate enabled, bqml_arima_xreg enabled,
automl disabled); execution `local_cpu`. Validates via 1.2.

### 1.4 `src/geaptimes/data/queries.py` + `utils/logger.py`
Parameterized SQL builders `build_source_query()` / `build_prepped_query()` from `DataConfig`
(top-N stations → daily agg + engineered covariates → per-station date spine + zero-fill → GSOD
weather + station metadata + calendar features → splits). Minimal logger. String-level tests.

### 1.5 `data_notebooks/01_citibike_prep.ipynb`
Loads config, builds `_source` then `_prepped` via `queries.py`. Verifies frozen-table max date,
LaGuardia GSOD ids, `citibike_stations` name-join coverage; visualizes ranges + split bands.

### 1.6 `scripts/setup_gcp.py`
Idempotent GCS bucket + BQ dataset provisioning from `ProjectConfig`; `--config` / `--dry-run`
flags; no-op if resources exist.

### 1.7 Notebook kernel standardization + cloud validation
Standardized `geaptimes` ipykernel (registered via `ipykernel install`; notebook pins
`kernelspec.name=geaptimes`); `.env.example` with `GCP_PROJECT=hybrid-vertex`; README + standards
updated (incl. opt-in Colab path). Read-only cloud validation against `hybrid-vertex` recorded in
`docs/notes` (frozen window 2013-07-01→2018-05-31; GSOD LaGuardia confirmed; 6/25 stations
unmatched on name-join — refinement to join metadata on `station_id` flagged).

### 1.8 Data-layer hardening + end-to-end build (hybrid-vertex)
Built `_source`/`_prepped` in `hybrid-vertex` (~4.4 GB billed). Fixes from the live build:
metadata join on `station_id` (CAST to STRING); `bq_location: US` (public data is US-multiregion —
single-region datasets can't read it); output table names moved to config (`data.source_table_name`
/`data.prepped_table_name`); `gender`/`usertype` are STRING (`gender='male'`); exclude empty-string
station name. **Result: 25 series, 18/25 with capacity** (7 absent from current snapshot →
keep + null capacity, do not drop). All facts recorded in `docs/notes`.

### 1.9 GCP resource labels + README badges
Every label-capable GCP asset carries `solution=geaptimes` via `geaptimes.constants.RESOURCE_LABELS`
(`bq_labels_option()` for DDL): applied in `gcp.ensure_dataset`/`ensure_bucket` and the table DDL;
back-applied to the existing `hybrid-vertex` dataset/bucket/2 tables; rule documented in
`CODE_STANDARDS.md`. README gets a centered badge banner (Python/uv/Ruff/ty/Pydantic/BigQuery/
Vertex AI/TimesFM).

### 1.10 Table naming convention (`geaptimes.naming`)
Human-readable token slugs: shared `source` → `__top{N}`; `prepped`/`train`/`infer` →
`__top{N}_h{horizon}_t{test}_v{validate}`. `table_names(cfg)` is the single source of truth;
`config_fingerprint(cfg)` is stamped into each table's description for traceability (tokens don't
capture covariate/weather/holiday changes — disambiguate via `experiment_name`). Notebook uses it;
live `hybrid-vertex` tables renamed to the slugged names + descriptions stamped. Documented in
`CODE_STANDARDS.md`. (train/infer builders land in Stage 2.)

### Stage 1 verification
- **No-cloud:** `uv sync` → `uv run ruff check .` → `uv run ruff format --check .` →
  `uv run ty check` → `uv run pytest` → load `base_config.yaml` via `ExperimentConfig.from_yaml`.
- **Cloud (manual, needs auth):** `setup_gcp.py --dry-run` then live; run the prep notebook.
- **STOP** checkpoint: present SQL logic, YAML+Pydantic schema, and table/split output.
