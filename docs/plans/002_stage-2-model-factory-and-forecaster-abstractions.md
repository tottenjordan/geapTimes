# geapTimes — Stage 2 Plan: Model Factory & Forecaster Abstractions

> Archived snapshot of the approved Stage 2 plan, frozen at approval. Immutable — see
> [`PLANS.md`](../../PLANS.md) and git for the as-built record.

## Context
Stage 1 delivered the typed config contract (`geaptimes.schemas`), the BigQuery data layer
(`geaptimes.data.queries`), deterministic table naming (`geaptimes.naming`), and a live prepped
table on `hybrid-vertex`. Stage 2 builds the **model layer**: a single `Forecaster` abstraction
plus a `ForecastFactory` that turns a validated `ModelConfig` into the right backend, for the
three targets in `base_config.yaml` — **TimesFM 2.5** (in-process foundation model), **BQML
ARIMA_PLUS_XREG** (server-side in BigQuery), and **Vertex AutoML** (managed training job).

The point of the abstraction is uniformity for later stages: Stage 3 logs runs and Stage 5
evaluates them, so every backend must return the **same prediction shape** despite radically
different execution models. The deferred Stage 1.10 item (`train`/`infer` table builders) also
lands here, since the cloud backends consume those tables.

## Locked decisions (from discussion)
- **Scope:** all three backends are **code-complete but NOT executed** in Stage 2 — no model
  downloads, no BQ jobs, no Vertex spend. First live runs happen in Stage 3 under experiment
  tracking. Verification is lint + type + unit tests only (offline, mocked).
- **Testability consequence:** forecasters must import and unit-test without loading the 200M
  checkpoint or touching GCP. TimesFM model load is **lazy** (deferred to `fit`/`predict`); BQ
  and Vertex clients are **injected** (default-constructed lazily, replaced by fakes in tests).
- **Factory name:** `ForecastFactory` (not `ForecasterFactory`).
- **Demo:** `data_notebooks/02_timesfm_local.ipynb` + thin `scripts/demo_timesfm.py`, authored
  and runnable but **first executed in Stage 3** (not run as part of Stage 2 verification).
- **Real TimesFM 2.5 API** (verified against installed `timesfm==2.0.1`):
  `m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo_id)`;
  `m.compile(timesfm.ForecastConfig(max_context=…, max_horizon=…, normalize_inputs=True,
  use_continuous_quantile_head=…, fix_quantile_crossing=…))`;
  `point, quantiles = m.forecast(horizon, inputs=[np.ndarray, …])`. Covariates via
  `forecast_with_covariates` remain **opt-in/deferred** (`TimesFMParams.use_covariates=False`).

## Common contract
```python
# models/base.py
QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]      # standardized levels emitted by every backend

@dataclass(frozen=True)
class ForecastResult:
    predictions: pd.DataFrame   # long: series, date, horizon_step, forecast, q10, q30, q50, q70, q90, model
    metadata: dict              # backend artifacts: device/checkpoint | bq_model_id | job/display_name

class Forecaster(ABC):
    name: str                   # model_cfg.name
    def __init__(self, model_cfg: ModelConfig, cfg: ExperimentConfig) -> None: ...
    @abstractmethod
    def fit(self) -> None: ...          # no-op for TimesFM (zero-shot)
    @abstractmethod
    def predict(self) -> ForecastResult: ...
```
All backends emit the same columns so Stage 5 evaluation is backend-agnostic. Each reads its
natural input: TimesFM pulls context arrays from the prepped table into pandas; BQML/AutoML
reference the `train`/`infer` BQ tables directly.

## Stage 2 — Detailed Items (Tier 2)

### 2.1 `models/base.py` — Forecaster ABC + ForecastResult
`Forecaster` ABC, `ForecastResult` dataclass, `QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]`, and the
canonical prediction column names (`q10, q30, q50, q70, q90`) as module constants (reused by
every backend + tests). No backend logic here.
Tests: ABC cannot be instantiated; a trivial concrete subclass satisfies the interface.

### 2.2 `data/queries.py` — `train` / `infer` builders (closes deferred 1.10)
Add `build_train_query(data, prepped, destination)` and `build_infer_query(data, prepped,
destination)`, both emitting `CREATE OR REPLACE TABLE … OPTIONS(labels=…) AS` (reuse
`bq_labels_option()`):
- **train** = rows where `splits != 'TEST'` (TRAIN+VALIDATE) — the fitting window.
- **infer** = the held-out forecast horizon: for this frozen dataset that is the `TEST` split
  (the most recent `splits.test_length` days), with the target nulled so the row schema matches a
  real inference request and the known-future regressors (temp/prcp/calendar) are genuinely
  present. Production future-dates-past-max (no weather actuals) are deferred to Stage 4.
Wire into `geaptimes.naming.table_names` (already returns `train`/`infer` keys). String-level
tests assert split filter, label OPTIONS, and the nulled target.

### 2.3 `models/timesfm.py` — `TimesFMForecaster` (in-process, device-agnostic)
- Lazy `_load_model()`: `TimesFM_2p5_200M_torch.from_pretrained(params.checkpoint)` then
  `compile(ForecastConfig(max_context=params.max_context, max_horizon=forecast.horizon,
  normalize_inputs=True, use_continuous_quantile_head=params.quantiles,
  fix_quantile_crossing=params.quantiles))`.
- Device: map `execution.target` (`local_gpu`→cuda / `local_cpu`→cpu) with
  `torch.cuda.is_available()` fallback; record chosen device in metadata.
- `fit()` = no-op (zero-shot). `predict()`: load context per series from the prepped table
  (last `context_len` points of the non-TEST window, via an injected loader), call
  `forecast(horizon, inputs=[…])`, map point→`forecast` and select the model's decile outputs
  for the standardized `QUANTILES` → `q10/q30/q50/q70/q90`, assemble the long `ForecastResult`.
- Inject the model loader + dataframe loader so tests use a fake returning fixed arrays — no
  checkpoint download, no BQ. Tests assert output schema, horizon length, device selection, and
  quantile channel mapping against the fake.

### 2.4 `models/bqml.py` — `BQMLForecaster` (server-side SQL)
- SQL builders: `build_create_model_sql()` → `CREATE OR REPLACE MODEL … OPTIONS(model_type=
  'ARIMA_PLUS_XREG'|'ARIMA_PLUS', time_series_timestamp_col, time_series_data_col,
  time_series_id_col, data_frequency='DAILY', holiday_region, labels=[('solution','geaptimes')])
  AS SELECT …` (XREG selects the `available_at_forecast_columns`); `build_forecast_sql()` →
  `ML.FORECAST` (XREG joins the `infer` table for future regressors).
- `fit()` submits the CREATE MODEL job; `predict()` runs ML.FORECAST and maps `forecast_value`
  (+ Gaussian `standard_error` → deciles via fixed z-scores) → standardized columns. BQ client
  injected; tests exercise SQL strings + result-frame mapping with a fake — no BQ execution.

### 2.5 `models/automl.py` — `AutoMLForecaster` (managed job config)
- Translate config → `AutoMLForecastingTrainingJob` constructor + `.run()` + `batch_predict`
  kwargs: column roles from `CovariateRoles`, `forecast_horizon`, `context_window`,
  `data_granularity_unit='day'`, `predefined_split_column_name='splits'`, `quantiles=QUANTILES`,
  `holiday_regions`, `labels={'solution':'geaptimes'}`.
- `fit()` creates a TimeSeriesDataset + launches training via the injected `aiplatform` seam;
  `predict()` batch-predicts over the `infer` table and maps the output. Gated/not executed in
  Stage 2. Tests assert the kwargs mapping + fit wiring (fake aiplatform) and the output mapping.

### 2.6 `models/factory.py` — `ForecastFactory`
`ForecastFactory.create(model_cfg, cfg) -> Forecaster` dispatching on `model_cfg.params.type`
(`"timesfm"|"bqml"|"automl"`), mirroring the schema's discriminated union;
`ForecastFactory.from_config(cfg) -> list[Forecaster]` building all **enabled** models. Unknown
type raises. Tests: each type → correct class; disabled models skipped; bad type raises.

### 2.7 Demo — `02_timesfm_local.ipynb` + `scripts/demo_timesfm.py`
Thin, reusing the package: load `base_config.yaml`, `ForecastFactory.create` the TimesFM model,
`predict()`, plot forecast vs TEST actuals with quantile band. `geaptimes` kernel pinned.
Authored + runnable; **not executed** in Stage 2 verification (first run in Stage 3).

## Verification (offline only — per "none run")
- `uv run ruff check .` → `uv run ruff format --check .` → `uv run ty check` → `uv run pytest`.
- `GCP_PROJECT=hybrid-vertex uv run python -c "from geaptimes.models.factory import
  ForecastFactory; from geaptimes.schemas import ExperimentConfig;
  fs=ForecastFactory.from_config(ExperimentConfig.from_yaml('config/base_config.yaml'));
  print([f.name for f in fs])"` → lists enabled forecasters (constructs all; no
  download/BQ/Vertex triggered because load is lazy).
- No live model load, no BQ, no Vertex. STOP checkpoint: present the abstraction, factory
  dispatch, the three backend wrappers, and the train/infer builders.

## Out of scope (later stages)
- Live execution + experiment logging (Stage 3); KFP pipelines + TimesFM cloud-inference
  container + comparison DAG (Stage 4); standardized MAE/RMSE/MAPE/Quantile-Loss eval +
  `run_experiment.py` CLI (Stage 5); TimesFM XReg covariates (opt-in, deferred).
