# geapTimes — Stage 4 Plan: Managed Pipelines & Cloud Orchestration

> Archived snapshot of the approved Stage 4 plan, frozen at approval. Immutable — see
> [`PLANS.md`](../../PLANS.md) and git for the as-built record.

## Context

Stages 1–3 delivered the typed config contract, the BigQuery data layer, the model factory with
three forecaster backends (TimesFM 2.5 in-process, BQML `ARIMA_PLUS_XREG`, AutoML wired/default-off),
and the Stage 3 **in-process** experiment harness (`run_experiment`: DOE → fit → predict → MAE/RMSE →
Vertex Experiments + GCS artifacts). Everything runs from a single local process today.

Stage 4 lifts that orchestration **into Vertex AI as a managed KFP v2 pipeline** and adds the first
**cloud-serving** path: a custom-container TimesFM model registered in Vertex Model Registry and
served either via an **online endpoint** (default) or **batch prediction**. The deliverable is a
**side-by-side comparison DAG** that runs all three backends (TimesFM + BQML + AutoML) on identical
backtests, emits a comparison artifact + winner, and tracks each model as a Vertex `ExperimentRun`
under `citibike-daily-baseline`. As with prior stages, every cloud seam is injected so the logic
unit-tests offline; the live `PipelineJob` (which incurs **one budgeted AutoML run** at the floor
budget + a transient endpoint deploy) is a separate, explicit verification step.

## Locked decisions

From discussion (user-owned calls):
- **TimesFM cloud inference**: build **both** custom-container paths — online **endpoint (default)**
  and **batch prediction** — selectable via `pipeline.serving.mode`.
- **Cost posture is a config switch** (`pipeline.serving.keep_deployed`): default **transient**
  (deploy → run → save artifacts → undeploy + delete); opt-in **leave-running**.
- **Comparison DAG includes all three backends.** AutoML runs live at the **floor budget**
  (`budget_milli_node_hours = 1000` = 1 node-hour; confirmed min via Google docs; already our default).

Mechanics / engineering defaults (my calls, stated for the record):
- **KFP v2** (`kfp>=2,<3`; aiplatform 1.158 does **not** pin kfp, so we choose it), compiled via
  `kfp.compiler.Compiler`, submitted as `aiplatform.PipelineJob`. **No** `google-cloud-pipeline-
  components` — thin `@dsl.component` shells over our existing forecasters are DRYer.
- **One `geaptimes` runtime container image** (Artifact Registry, labelled `solution=geaptimes`),
  used both as the TimesFM **serving container** and as the **base_image** for heavy KFP components
  (KFP overrides the entrypoint, so the serving CMD is inert in components). Built via **Cloud Build**
  (`gcloud builds submit`) to avoid local docker + corp airlock.
- **TimesFM checkpoint baked into the image** at build time (safer behind VPC-SC than a runtime HF
  download from a locked-down serving node).
- **Served** TimesFM predictions (endpoint/batch per mode) feed the comparison — the cloud-inference
  path is genuinely part of the DAG, not bolted on; in-process TimesFM stays available via Stage 3.
- Comparison runs reuse experiment **`citibike-daily-baseline`** (cross-stage comparable).
- Predictions passed between components as **Parquet** on `gs://{pipeline_root}/…`; `dsl.ExitHandler`
  wraps serve+teardown so the endpoint is never stranded on partial failure.
- Same injected-seam / offline-test discipline as Stages 2–3; `base_config.yaml` stays cheap
  (AutoML disabled); a committed `config/comparison_config.yaml` defines the all-three live run.

## Stage 4 — Detailed Items (Tier 2)

### 4.1 Schema + config: `pipeline:` block
- `schemas.py`: add `ArtifactRegistryConfig` (repo/image_name/image_tag/location → resolved image
  URI), `ServingConfig` (`mode: Literal["endpoint","batch"]="endpoint"`, `keep_deployed=False`,
  machine/replica knobs for endpoint + batch), `PipelineConfig` (`pipeline_root: str|None`, `image`,
  `serving`, `component_machine_type`, `enable_caching=True`). Add
  `pipeline: PipelineConfig = Field(default_factory=PipelineConfig)` to `ExperimentConfig`.
- `config/base_config.yaml`: append a `pipeline:` block with defaults; **do not** flip `automl.enabled`.
- `config/comparison_config.yaml` (new): all three models enabled, `automl.budget_milli_node_hours:
  1000`, `execution.experiment_name: citibike-daily-baseline`, `execution.target: cloud_pipeline`.
- Tests: pipeline defaults; image-URI + pipeline_root derivation; invalid `mode` rejected; both YAMLs
  validate.

### 4.2 Factor the TimesFM core (behavior-preserving refactor)
- New `models/timesfm_core.py`: `compile_timesfm(model, *, max_context, horizon, quantiles)`,
  `forecast_arrays(model, inputs, horizon) -> (point, quantiles)`, `rows_from_forecast(series_ids,
  last_dates, point, quantiles, *, horizon, model_name)` (keep `_quantile_channels`).
- Refactor `models/timesfm.py` `predict()` to use them (load frame → build inputs → `forecast_arrays`
  → `rows_from_forecast`); existing `tests/test_timesfm.py` must stay green. New `test_timesfm_core.py`.

### 4.3 TimesFM serving predictor + app
- `pipelines/serving/predictor.py`: `TimesFMPredictor(*, model_loader, params)` — load+compile once;
  `predict(instances) -> responses` reusing the factored core (emits point + q10..q90 arrays).
- `pipelines/serving/app.py`: web app honoring the Vertex contract — `GET {AIP_HEALTH_ROUTE=/health}`,
  `POST {AIP_PREDICT_ROUTE=/predict}` on `0.0.0.0:{AIP_HTTP_PORT=8080}`; reads checkpoint/context/
  horizon/quantiles from env. Request `{"instances":[{"series_id","context":[…]}]}` → response
  `{"predictions":[{"series_id","horizon","point":[…],"q10":[…],…,"q90":[…]}]}`.
- Tests: predictor with a **fake model** (deterministic `(point,quantiles)` arrays) — response shape,
  array length == horizon, channel mapping (q50 = channel 5), context slicing to `context_len`; app
  via Flask/FastAPI test client with an injected fake predictor (no real server, no torch).

### 4.4 Finish the AutoML predict path
- `models/automl.py`: replace the `NotImplementedError` in `_read_predictions()` — after
  `batch_predict`, resolve the BQ output table from the job's output info and `SELECT *` via an
  injected BigQuery runner (lazy default). Add `_flatten_automl_output(raw)` (run before the existing
  `_to_predictions`): `value` ← `predicted_<target>.value`; per `QUANTILES` level, match in
  `quantile_values` and pull the parallel `quantile_predictions[idx]` into `qXX`; series/date pass
  through. Keep the `prediction_reader` seam so the current test stays valid.
- Tests: `_flatten_automl_output` against a fake nested-struct frame (structs→dicts, arrays→lists).
  Exact live field names confirmed in 4.9 (Stage-3 BQML-quirk discipline: adjust + note if different).

### 4.5 Pipeline config helpers + step functions (pure, injected, offline)
- `pipelines/config.py`: pure derivations off `ExperimentConfig` (`image_uri`, `pipeline_root`,
  endpoint/model display names), label-stamped.
- `pipelines/steps.py`: the real logic as plain functions with injected seams — `build_tables_step`,
  `run_backend_step` (reuses `ForecastFactory` + `point_metrics` + `ExperimentTracker`),
  `register_timesfm_step`, `deploy_endpoint_step`/`undeploy_endpoint_step`, `endpoint_predict_step`,
  `batch_predict_timesfm_step`, `compare_step` (winner = lowest RMSE, tie-break MAE).
- Tests: each step with fakes (query_runner, forecasters, aiplatform, endpoint client); `compare_step`
  winner/tie-break. Reuses Stage-3 fake patterns.

### 4.6 KFP components + DAG + compile  *(riskiest)*
- `uv add 'kfp>=2,<3'`; pin resolved version.
- `pipelines/components.py`: `@dsl.component(base_image=IMAGE)` shells that reconstruct cfg from a
  JSON pipeline param (`model_validate_json`), call the 4.5 steps, emit `Output[Dataset]`/`Metrics`.
- `pipelines/pipeline.py`: `@dsl.pipeline` assembling build_tables → {timesfm, bqml, automl} → compare;
  `dsl.Condition` on `serving.mode` (endpoint vs batch) and `keep_deployed`; `dsl.ExitHandler` around
  serve+teardown.
- `pipelines/compile.py`: `compile_pipeline(out_path)` (pure, no cloud).
- Tests: compile to a temp YAML; assert component set, dependency edges, and the endpoint/batch +
  keep_deployed condition branches.

### 4.7 Runtime container + Cloud Build  *(riskiest)*
- `pipelines/container/Dockerfile` (python:3.11-slim + `uv sync` + package + baked TimesFM checkpoint;
  serving CMD runs `app.py`), `pipelines/container/cloudbuild.yaml` (build + push to AR, labelled).
- Regenerate `requirements.txt` (`uv export --no-hashes --no-dev`); extend `scripts/setup_gcp.py` to
  ensure the labelled Artifact Registry repo.
- Mitigate torch image size (CPU-only torch, slim base, layer caching); record build time + digest.

### 4.8 Submit + AutoML-enable override
- `pipelines/submit.py`: `submit_pipeline(cfg, *, aiplatform)` → compile + `PipelineJob(...,
  experiment=cfg.execution.experiment_name).submit()` (aiplatform injected). `--enable-automl` flag
  applies a DOE-style `model_dump` → patch → `model_validate` override (no in-place mutation).
- Tests: submit with a fake aiplatform (asserts compile + PipelineJob kwargs + labels); override
  re-validates and enables AutoML.

### 4.9 Live run + notes + PLANS  *(riskiest; incurs spend)*
- Confirm AR repo + Cloud Build API reachable; `gcloud builds submit` → labelled image (record digest).
- `submit.py` with `comparison_config.yaml` on `hybrid-vertex`. Confirm: `PipelineJob` SUCCEEDED;
  three backend `ExperimentRun`s under `citibike-daily-baseline`; TimesFM endpoint deployed then
  **torn down** (transient mode); one AutoML job at floor budget; comparison artifact + winner in
  `gs://…/pipeline_root/…`; all assets labelled `solution=geaptimes`.
- Record run id, metrics, winner, AutoML output-schema confirmation, image digest, timings/cost, and
  any quirks in `docs/notes/stage-4-managed-pipelines-live-run.md`. Update PLANS.md Tier 2 statuses +
  hashes and Tier 1 Stage 4 → done.

## Files

- **New**: `src/geaptimes/models/timesfm_core.py`; `src/geaptimes/pipelines/{__init__,config,steps,
  components,pipeline,compile,submit}.py`; `src/geaptimes/pipelines/serving/{__init__,predictor,app}.py`;
  `src/geaptimes/pipelines/container/{Dockerfile,cloudbuild.yaml}`; `config/comparison_config.yaml`;
  `docs/notes/stage-4-managed-pipelines-live-run.md`; `docs/plans/004_stage-4-managed-pipelines-and-
  cloud-orchestration.md`; tests `test_timesfm_core.py`, `test_serving_predictor.py`,
  `test_serving_app.py`, `test_pipeline_config.py`, `test_pipeline_steps.py`, `test_pipeline_compile.py`,
  `test_pipeline_submit.py`.
- **Edit**: `src/geaptimes/schemas.py` (pipeline block); `config/base_config.yaml` (pipeline block);
  `src/geaptimes/models/timesfm.py` (use core); `src/geaptimes/models/automl.py` (predict path);
  `scripts/setup_gcp.py` (AR repo); `pyproject.toml`/`uv.lock` (`kfp`); `requirements.txt`;
  `tests/test_schemas.py`, `tests/test_timesfm.py`, `tests/test_automl.py`; `PLANS.md`.
- **Reuse**: `geaptimes.models.factory`/`base`/`bqml`/`automl`, `geaptimes.experiment.{metrics,
  tracking,runner}`, `geaptimes.naming.table_names`, `geaptimes.data.queries` (train/infer builders),
  `geaptimes.constants.RESOURCE_LABELS`, `geaptimes.gcp`.

## Verification

- **Offline** (gating): `uv run ruff check .` → `uv run ruff format --check .` → `uv run ty check`
  → `uv run pytest`. Then `compile_pipeline()` emits the DAG YAML with no cloud calls, and
  `ExperimentConfig.from_yaml('config/comparison_config.yaml')` validates (all three enabled).
- **Live** (separate, auth): Cloud Build the image → submit the comparison `PipelineJob` on
  `hybrid-vertex` → confirm success, the three `ExperimentRun`s, transient endpoint teardown, the
  floor-budget AutoML run, and the comparison artifact + winner; record in `docs/notes/`.
- **STOP** checkpoint: present the schema/config switches, the factored TimesFM core + serving
  container/predictor, the finished AutoML predict path, the KFP components + comparison DAG, the
  submit path, and the live comparison results (metrics + winner + cost-safe teardown).

## Out of scope (Stage 5)

Standardized eval suite (MAE/RMSE/MAPE/Quantile Loss) + comparison report; the polished
`scripts/run_experiment.py` CLI; TimesFM XReg covariates; leave-running endpoint as a default;
a dedicated `04_*` demo notebook (add only if requested).
