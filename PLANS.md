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
| 3 | Experiment Tracking & DOE Framework | done |
| 4 | Managed Pipelines & Cloud Orchestration | done |
| 5 | Standardized Evaluation & Entry Point | done |

**Stage 2** — `src/geaptimes/models/`: `base.py` (ABC), `factory.py`, `automl.py`,
`timesfm.py` (CPU/GPU), `bqml.py`; local TimesFM inference demo.
**Stage 3** — DOE matrix engine in `utils/`; wire `aiplatform.Experiment`/`ExperimentRun`;
per-run GCS path `gs://<bucket>/experiments/<name>/<timestamp>/`; log params + metrics.
**Stage 4** — KFP pipelines in `src/geaptimes/pipelines/`; TimesFM cloud-inference custom
container (Model Garden → endpoint); side-by-side comparison DAG.
**Stage 5** — uniform eval (MAE/RMSE/MAPE/Quantile Loss) over identical backtests; finalize
`scripts/run_experiment.py` CLI.

---

## Tier 2 — Stage 5 (Standardized Evaluation & Entry Point) — COMPLETE

Approved 2026-06-30; snapshot `docs/plans/006_stage-5-standardized-evaluation-and-entry-point.md`.
Delivers the standardized evaluation suite (MAE / RMSE / **sMAPE** / **quantile loss** + per-series
breakdown) over identical backtests from **one** metric implementation shared by the offline runner
and the pipeline scorer, plus the polished `scripts/run_experiment.py` entry point. One deferred item
is folded in by user decision: **warm-endpoint reuse** (`keep_deployed` + fingerprint-label reuse +
teardown guard); the **optional-quantiles** switch stays in the backlog (`CODE_STANDARDS.md`).

| # | Item | Status | Commit |
|---|------|--------|--------|
| 5.1 | `experiment/metrics.py` — eval suite: guarded sMAPE, pinball/quantile loss over `QUANTILES`, per-series breakdown, `evaluate()` aggregator (keep `point_metrics`) | done | `5d71aa8` |
| 5.2 | Unify: wire `evaluate()` into `runner.py` + `score_and_track_step`; shared `rank_backends` helper for `compare_backends` Markdown ranking | done | `7244b3c` |
| 5.3 | `scripts/run_experiment.py` CLI — argparse over `run_experiment` (`--config`, `--enable/--disable-automl`, `--dry-run`), RunRecord table + comparison report | done | `518a1b9` |
| 5.4 | Warm endpoints + reuse: `keep_deployed` warm mode, `find_reusable_endpoint_step` (fingerprint = sha256(image **digest** + serving env) label lookup), DAG reuse branch, "only tear down what this run created" guard | done | `a2f5e1f` |
| 5.5 | Offline gate + cheap `--disable-automl` live run (warm deploy → reuse → guarded teardown + unified richer metrics live); `docs/notes/`; **STOP** | done | `1d0fb7d` |

### Stage 5 design notes

- **sMAPE over MAPE** — targets are daily station counts with zero-fill; raw MAPE divides by ~0, so
  symmetric MAPE with a small-denominator guard (documented) is the interpretable choice.
- **One metric implementation** — the suite lives in `metrics.py` and is consumed by both the offline
  `runner.py` and the pipeline `score_and_track_step`/`compare_backends`, so "identical backtests" is
  enforced by construction, not convention.
- **One live run, not two** — warm-endpoint reuse only touches TimesFM serving, so a cheap
  `--disable-automl` run validates both the reuse path and the unified richer scorer live; AutoML's
  richer-metrics path is covered offline + by the already-proven train/infer split.
- **Reviews-as-gates** ([[sdd-reviews-as-gates]]) on 5.4 — the teardown guard + concurrency are the
  riskiest surface (deleting a shared warm endpoint is a destructive regression).

**5.4 implementation notes (2026-06-30, `a2f5e1f`):** snapshot `docs/plans/006_*` covers the design;
key build decisions: (1) reuse is resolved **at submit time** (a live call sequence like the AutoML
preflight) and the DAG **branches at compile time** — not a runtime `dsl.If` — matching the recorded
"branch at compile time, the YAML shows exactly what executes" convention; `build_pipeline(cfg, *,
reused_endpoint, serving_fingerprint)` are compile-time inputs, not pipeline params. (2) Digest
resolution shells `gcloud` behind an injected runner (the 4.7 injectable-gcloud precedent), not a new
AR SDK dep. (3) Reuse predict is a small sibling component `endpoint_predict_by_name` (a plain string
param), avoiding the importer-URI semantics that bit us in P2.7a. **Two teardown guards:** primary —
no teardown task is emitted on a reuse run (`needs_teardown` gate); secondary —
`teardown_serving_step` skips any endpoint/model labelled `keep=true`. Reviews-as-gates
([[sdd-reviews-as-gates]]): code-quality reviewer APPROVE-WITH-NITS; fixed the one MEDIUM finding
before the gate (the `keep` label is now stamped whenever `keep_deployed`, **independent of reuse
mode**, so a warm endpoint created without reuse is still protected from a later transient run sharing
its display name) plus a fail-loud on an empty resolved digest. Offline gate green (218 passed).
**Live validation deferred to 5.5.**

**5.5 live validation (2026-07-01):** full writeup in
`docs/notes/stage-5-warm-endpoint-reuse-live-run.md`. Cheap `--disable-automl` sequence on
`hybrid-vertex`: **warm deploy** (`…20260630232242`, SUCCEEDED) left one endpoint labelled
`fingerprint=82717760…` + `keep=true` with no teardown node emitted; **reuse** (`…20260701024425`,
SUCCEEDED) resolved that endpoint at submit time and compiled the reuse branch
(`endpoint-predict-by-name`, no serving lifecycle, no teardown) — predicting against the warm
endpoint with **identical** timesfm metrics (MAE 85.51 / RMSE 113.65), proving reuse correctness;
**guard probe** ran `teardown_serving_step` live and skipped both the `keep=true` endpoint and model
(destructive-regression guard confirmed); **cleanup** deleted both by resource name. The unified
richer suite (sMAPE 39.26 / quantile loss 31.30 for timesfm; 42.66 / 30.27 for bqml) flowed through
the pipeline scorer live — `NaN` on the pre-5.2 06-26 runs, populated now. **Gotcha:** the first
submit failed because the runtime image predated 5.4 — a component rejected the unknown
`reuse_endpoint` field (`extra=forbid`); a schema change needs an image rebuild (Cloud Build
`80355d6b`, digest `cc3a5c00…`) before any live run. Offline gate green (218 passed).

### Post-Stage-5 — eval-metric comparability audit + security hooks (2026-07-01)

Snapshot `docs/plans/007_post-stage5-eval-audit-and-tooling.md`. Two independent workstreams off the
Stage-5 backlog, prompted by "AutoML eval metrics haven't looked comparable" + backlog #3.

| WS | Item | Status | Commit |
|----|------|--------|--------|
| A | Eval-metric comparability audit + hardening (`comparison.py` n_points column + parity `warnings` + NaN-safe ranking; `metrics._merge` partial-overlap WARNING; tests; `docs/notes/automl-eval-metric-comparability.md`) | done | `d02f044` |
| B | Backlog #3 modern-python security hooks (`detect-secrets` active + committed `.secrets.baseline`; `shellcheck`/`actionlint`/`zizmor` pinned but dormant; `CODE_STANDARDS.md`) | done | `3c5c737` |

- **Audit conclusion (WS A):** the shared scorer is structurally fair — every backend funnels into one
  `evaluate()` (inner join on `(series, date)` vs TEST actuals) → `rank_backends`. Confirmed live that
  AutoML scored over `n_points=308`, **identical** to TimesFM/BQML → **no scoring/alignment bug**; the
  AutoML MAE gap (142 vs 78–85) is real underperformance at the floor budget. The hardening makes
  non-comparability *loud* (differing point counts warn; `n_points` on the table; NaN can't corrupt the
  sort) rather than changing any metric value. **Still open:** AutoML's full sMAPE/quantile-loss suite
  has **never** been produced live (all successful AutoML runs predate the 5.2 scorer; 5.5 ran
  `--disable-automl`) — a full 3-backend confirmation run (~2.5 h + AutoML budget) would close it, offered
  separately, not required by A/B.
- **Backlog dispositions (declined this session):** **#2 `dsl.ParallelFor`** — deferred; the backends
  are a heterogeneous compile-time loop, reserve ParallelFor for a homogeneous inner axis (DOE/backtest
  windows); rationale `docs/notes/pipeline-parallelfor-deferred.md`. **#4 AutoML tabular workflow /
  `AutoMLForecastingTrainingJobRunOp`** — held; far costlier than our locked floor-budget single job and
  fragments the AutoML logic, and the preflight already de-risks the raw SDK call.

---

## Tier 2 — Stage 4 (Managed Pipelines & Cloud Orchestration) — COMPLETE

The Stage 3 in-process `run_experiment` loop is lifted into a managed **KFP v2 pipeline** on Vertex
AI, and a custom-container **TimesFM** model is served via an online **endpoint** (default) or
**batch** prediction (selectable via `pipeline.serving.mode`; teardown is the `keep_deployed`
switch, default transient). The deliverable is a **side-by-side comparison DAG** running all three
backends (TimesFM + BQML + AutoML at the floor budget) on identical backtests, emitting a comparison
artifact + winner and one Vertex `ExperimentRun` per model under `citibike-daily-baseline`. One
`geaptimes` runtime image (Artifact Registry, built via Cloud Build) serves double duty as the
TimesFM serving container and the KFP component base image. Every cloud seam is injected so logic
unit-tests offline (incl. a pipeline-compile graph test); the live `PipelineJob` — which incurs one
budgeted AutoML run + a transient endpoint deploy — is a separate, explicit step. Full standardized
evaluation + the polished CLI remain in Stage 5.

| # | Item | Status | Commit |
|---|------|--------|--------|
| 4.1 | `schemas.py` + `base_config.yaml` + `comparison_config.yaml` — `pipeline:` block (serving mode/teardown, AR image, pipeline_root) | done | `d8ec2ae` |
| 4.2 | `models/timesfm_core.py` — factor TimesFM core (`compile_timesfm`/`forecast_arrays`/`rows_from_forecast`); refactor `timesfm.py` | done | `c43836e` |
| 4.3 | `pipelines/serving/{predictor,app}.py` — TimesFM serving predictor + Vertex-contract web app | done | `78ca456` |
| 4.4 | `models/automl.py` — finish predict path (`_read_predictions` output read + `_flatten_automl_output`) | done | `5f0323b` |
| 4.5 | `pipelines/{config,steps}.py` — pure pipeline helpers + injected-seam step functions | done | `a429bdb` |
| 4.6 | `pipelines/{components,pipeline,compile}.py` — KFP components + comparison DAG + compile; add `kfp` | done | `82cafda` |
| 4.7 | `pipelines/container/{Dockerfile,cloudbuild.yaml}` + `setup_gcp.py` AR repo + `requirements.txt` | done | `2802b12` |
| 4.8 | `pipelines/submit.py` — submit `PipelineJob` + `--enable-automl` override | done | `a8303ff` |
| 4.9 | Live comparison run on `hybrid-vertex` (TimesFM + BQML + AutoML floor) + notes | done (via redesign P2.7b) | `0df799c` |

### 4.1 Schema + config
`ArtifactRegistryConfig` + `ServingConfig` (`mode: endpoint\|batch`, `keep_deployed`, machine/replica
knobs) + `PipelineConfig` (`pipeline_root`, `image`, `serving`, `component_machine_type`,
`enable_caching`); `ExperimentConfig.pipeline`. `base_config.yaml` gets a `pipeline:` block (AutoML
stays disabled); new `comparison_config.yaml` enables all three at floor budget under
`citibike-daily-baseline` / `target: cloud_pipeline`.

### 4.2 Factor TimesFM core
Behavior-preserving extraction of compile + `model.forecast` + tensor→`PREDICTION_COLUMNS` mapping
into `models/timesfm_core.py`, reused by `TimesFMForecaster.predict()` and the serving predictor.

### 4.3 Serving predictor + app
`TimesFMPredictor` (load+compile once; `predict(instances)→responses` via the core) + a Vertex
custom-container web app (`/health`, `/predict`, `AIP_*` env). Offline-tested with a fake model +
test client (no torch, no server).

### 4.4 AutoML predict path
Implement the batch-prediction output-table read + nested-struct flatten
(`predicted_<target>.{value,quantile_values,quantile_predictions}` → `PREDICTION_COLUMNS`), keeping
the injected `prediction_reader` seam. Live field names confirmed in 4.9.

### 4.5 Pipeline helpers + steps
`config.py` pure derivations (image URI, pipeline_root, display names); `steps.py` plain step
functions (build tables, per-backend run via `ForecastFactory`+`point_metrics`+`ExperimentTracker`,
register/deploy/undeploy/endpoint-predict/batch-predict, compare→winner) with injected seams.

### 4.6 KFP components + DAG + compile
`@dsl.component` shells over the steps (cfg reconstructed from a JSON param); `@dsl.pipeline`
assembling build_tables → {bqml, automl in-process; timesfm served (endpoint|batch)} → compare;
`compile_pipeline()` (pure). Compile-to-YAML graph test (4 configs). Adds `kfp==2.16.1`.

**Refinements vs the approved plan (sound engineering calls, recorded):**
- *Branch at compile time, not via `dsl.Condition`.* Which backends run, the serving `mode`, and
  whether to tear down are read from the typed config when `build_pipeline(cfg)` runs — we compile
  per submit, so the cfg is known and the YAML shows exactly what executes (and is graph-testable).
  The rich config still travels to component bodies as a runtime `config_json` param.
- *`dsl.ExitHandler` wraps the **whole** body* (teardown as the exit task), not just serve+teardown:
  KFP forbids a task outside a handler depending on one inside it, so `compare` (which needs the
  served-TimesFM metrics) must sit inside the handler too. Teardown is config/display-name based and
  idempotent (no-op when nothing is deployed), so always-on-exit is safe.
- *steps.py additions* (same injected-seam discipline): `deploy_endpoint_step` now creates a
  named+labelled endpoint so teardown can resolve it by display name; `teardown_endpoint_step`
  (name-based) → `teardown_serving_step` (config-based, for the ExitHandler); new
  `score_and_track_step` gives the served path the same metrics/`ExperimentRun`/artifacts as
  in-process backends without an in-process `Forecaster`.

### 4.7 Container + Cloud Build
One `geaptimes` runtime image (baked TimesFM checkpoint, serving CMD) + `cloudbuild.yaml` (push to
labelled AR repo); regen `requirements.txt`; `setup_gcp.py` ensures the AR repo. **Offline files
done** (`2802b12`); the live `gcloud builds submit` (image build + digest record) runs in 4.9.

**Build-dependency decision (recorded):** the image installs **CPU-only torch** from the PyTorch CPU
index, then the project from **public PyPI** — deliberately *not* `uv.lock`, which pins a CUDA torch
(86 nvidia-* pkgs, ~6 GB) from an internal mirror (`artifact-foundry-prod`) that Cloud Build workers
can't reach. `timesfm[torch]` only needs `torch>=2.0.0`, so CPU satisfies it. AR repo provisioning
shells out to `gcloud` (injectable runner) rather than adding an admin SDK dep to the runtime image.

### 4.8 Submit + AutoML-enable override
`submit_pipeline(cfg, *, aiplatform)` compiles + submits a `PipelineJob` under the experiment;
`--enable-automl` flips AutoML on via a `model_dump`→patch→`model_validate` override.

### 4.9 Live run + notes
Cloud Build the image, submit the comparison `PipelineJob`; confirm success, three `ExperimentRun`s,
transient endpoint teardown, floor-budget AutoML run, and comparison artifact + winner; record in
`docs/notes/stage-4-managed-pipelines-live-run.md`.

### Stage 4 verification
- **Offline** (gating): `uv run ruff check .` → `uv run ruff format --check .` → `uv run ty check`
  → `uv run pytest`; then `compile_pipeline()` emits the DAG YAML and `comparison_config.yaml`
  validates — no cloud calls.
- **Live** (manual, auth): Cloud Build the image, submit the comparison `PipelineJob`, confirm
  success + teardown + AutoML floor run + comparison artifact, record in `docs/notes/`.
- **STOP** checkpoint: present the config switches, the factored core + serving container, the
  AutoML predict path, the KFP components + comparison DAG, the submit path, and the live results.

### Stage 4 follow-up feedback (triaged 2026-06-26)

Six pipeline-feedback points raised during the 4.9 live run, with disposition. None affect the
in-flight run (improvements show up on the next deploy):

| # | Feedback | Disposition |
|---|----------|-------------|
| 1 | Improve pipeline step names | **done now** — explicit `set_display_name` per task (`run-backend:<model>`, `build-tables`, …); compile test asserts them |
| 2 | Make quantiles optional (force `minimize-quantile-loss` + adjust metrics only when enabled) | **deferred → Stage 5**, documented in `CODE_STANDARDS.md` (Deferred / backlog) |
| 3 | Output artifacts for register-timesfm (model + markdown metadata summary) | **Stage 4 addendum** 4A.4/4A.6 (GCPC ops emit these natively) |
| 4 | Output artifacts for build-tables / run-backend / compare / deploy-endpoint | **Stage 4 addendum** 4A.2/4A.4 |
| 5 | Adopt Google Cloud Pipeline Components where beneficial (esp. serving lifecycle) | **Stage 4 addendum** 4A.6 — adopt GCPC for serving infra; keep custom components for logic+tracker steps; do **not** adopt the heavyweight AutoML tabular workflow |
| — | Data-prep as first pipeline step (create source/prepped if absent; outputs as artifacts) | **Stage 4 addendum** 4A.1/4A.3 (existence+fingerprint guard; self-bootstrapping pipeline) |
| 6 | Reuse an existing similar TimesFM deployment | **deferred → Stage 5** (coupled to warm/`keep_deployed` endpoints); design + LOE in `docs/notes/timesfm-endpoint-reuse-design.md` |

Items #3/#4/#5 (plus the data-prep-step idea below) are coupled (GCPC serving ops deliver much of
#3/#4's standardized artifacts + lineage for free) and are folded into the **Stage 4 addendum**
below, decided at the STOP checkpoint, so they land in one combined re-run.

### Stage 4 Addendum — Self-contained data prep, richer artifacts & GCPC (PROPOSED 2026-06-26)

> **SUPERSEDED 2026-06-26 by the Stage 4 Redesign** (approved; snapshot
> `docs/plans/005_stage-4-redesign-train-inference-split.md`). The first live run failed (AutoML read
> bug, fixed `e8a0f5f`), exposing that `run_backend` welds fit+predict+score+track into one pod. The
> redesign splits every backend into **train → infer → score** (trained model passed as an artifact;
> failed infer re-uses the cached model) and rolls out **phased**: Phase 1 = the split (custom, no new
> deps); Phase 2 = **hybrid** GCPC (serving lifecycle only) + the surviving 4A items (4A.1–4A.5, 4A.8).
> The 4A.1–4A.8 items below are retained for reference but are now executed via the redesign phases.

**Status: SUPERSEDED (see banner above).** Make the comparison pipeline
**self-bootstrapping** (no out-of-band Stage 1 notebook prerequisite) and **lineage-rich**, and
adopt **Google Cloud Pipeline Components (GCPC)** where they earn their keep. Folds feedback #3/#4/#5
and the data-prep-as-first-step idea into one change set, verified in **one combined re-run**
(iterate cheaply with `--disable-automl`; one full three-backend run to close). Archive an immutable
snapshot to `docs/plans/005_*.md` on approval.

Design principles (carried from Stage 4):
- **Keep custom `@dsl.component` shells** for steps with real logic + injected-seam offline tests +
  `ExperimentTracker` integration (data prep, build-tables, run-backend, served scoring, compare).
- **Adopt GCPC only for opaque managed-resource infra** (serving lifecycle) where it adds
  standardized `google.Vertex*` artifacts + ML-Metadata lineage + billing-label propagation and
  removes code we'd otherwise hand-roll. GCPC ops are opaque → test DAG wiring via compile, not
  internals.
- **Table artifacts are *references***, not copied bytes: a `dsl.Dataset` (or `google.BQTable`) with
  URI `bq://project.ds.table` and `.metadata` = {table id, `config_fingerprint`, row count}.

| # | Item | Notes |
|---|------|-------|
| 4A.1 | `ensure_source` → `ensure_prepped` steps at the front of the DAG | existence + `config_fingerprint` guard (skip when present & current); `CREATE OR REPLACE` only when missing/stale; injected `table_exists`/`table_metadata` seams for offline tests |
| 4A.2 | Table-reference artifacts | `ensure_source`/`ensure_prepped`/`build_tables` emit `Dataset` artifacts (`bq://…` + fingerprint + row count) — satisfies the data side of #4 |
| 4A.3 | Wire `prepped` artifact into consumers | `build_tables` **and** the TimesFM endpoint/batch steps take `prepped` as an explicit input — fixes the serving branch's current *implicit* (edge-less) dependency on prepped existing |
| 4A.4 | Results artifacts | `run_backend`/served → `Output[Metrics]` (MAE/RMSE scalars in console) + ExperimentRun name in metadata; `compare` → `Output[Markdown]` ranking; (serving model/endpoint artifacts come free from 4A.6) — satisfies #3 + results side of #4 |
| 4A.5 | Freshness override | `data.force_rebuild` config flag + `--force-data-rebuild` CLI so the guard can be bypassed on demand |
| 4A.6 | Adopt GCPC serving ops | `ModelUploadOp`, `EndpointCreateOp`, `ModelDeployOp`, `ModelUndeployOp`/`EndpointDeleteOp` (+ optional `ModelBatchPredictOp`); emit `google.VertexModel`/`VertexEndpoint` artifacts w/ lineage; `uv add google-cloud-pipeline-components` (pin) — satisfies #5 + serving side of #3/#4 |
| 4A.7 | Tests + docs + verification | compile-graph tests (new nodes/edges) + step unit tests (table-exists/metadata seams); record the GCPC adoption decision in `CODE_STANDARDS.md` + `CLAUDE.md`; offline gate; **cheap `--disable-automl` live test** of data step + artifacts + GCPC serving, then one full run |
| 4A.8 | Right-size `component_machine_type` | drop the default from `e2-standard-4` → `e2-standard-2` (config + `schemas.py` default). The component pods are **supervisors** (esp. `run-backend:automl`, which blocks polling the managed AutoML `TrainingPipeline`), not compute; 4 vCPU sits ~idle. Smaller pod = lower cost, no latency cost. Consider a per-task override if any in-process step (BQML/TimesFM serving glue) ever needs more |

**Notes / rationale:**
- *Frozen data → the data step is almost always a no-op.* Stage 1 pinned the public window to
  2018-05-31, so `source`/`prepped` are effectively static; the guard makes the common case a fast
  existence check. The value is reproducibility / fresh-project bootstrap, not per-run compute
  (a naive `CREATE OR REPLACE` would re-bill the ~4.4 GB source every run — explicitly avoided).
- *Two steps, not one* (`source` → `prepped`): `source` changes ~never while `prepped` changes with
  covariate/split config, so independent caching + cleaner lineage (source → prepped → {train,
  infer, timesfm}).
- *Coupling of #3/#4 with #5:* serving-step artifacts (model, endpoint) are delivered by the GCPC
  ops in 4A.6, so 4A.4 only hand-rolls artifacts for the **custom-kept** steps.
- *BigQuery DDL via `BigqueryQueryJobOp`?* Evaluated, **likely keep custom** — our builders carry the
  existence/fingerprint **skip** logic + offline tests that an opaque op can't; we can still emit a
  `google.BQTable` artifact from the custom component.
- *Component pods are supervisors, not workers (4A.8).* Live introspection of the comparison run
  (2026-06-26) showed the `run-backend:automl` component is a CustomJob on `e2-standard-4` that merely
  launches the managed AutoML `TrainingPipeline` (a separate resource) and then **blocks polling it**
  — so its CPU chart is flat by design, and the real trainer runs on Google-managed infra with no
  user-visible node metrics. AutoML itself exposes no machine/parallelism knob (only `budget_milli_
  node_hours`, already at the 1000 floor); the only addressable lever is shrinking the supervisor pod.

**Out of scope (held):** the heavyweight AutoML tabular workflow
(`get_automl_forecasting_pipeline_and_parameters`) — far costlier than our locked floor-budget single
job; and `AutoMLForecastingTrainingJobRunOp` — modest benefit, fragments the AutoML logic (column
specs + flatten + score + track) and our preflight already de-risks the raw SDK call.
*(Note: the `AutoMLForecastingTrainingJobRunOp` stance is being revisited in the Stage 4 redesign
discussion — train/inference split makes GCPC ops fit; see below.)*

**Backlog — `dsl.ParallelFor` (deferred 2026-06-26):** not for the cross-service axis (backends
already run in parallel; their chains are heterogeneous; conflicts with compile-time branch
resolution). Reserve for a homogeneous inner axis (DOE/HP variants, backtest windows). Full
rationale: `docs/notes/pipeline-parallelfor-deferred.md`.

---

## Tier 2 — Stage 4 Redesign (Train/Inference Split + Hybrid GCPC) — COMPLETE

Approved 2026-06-26; snapshot `docs/plans/005_stage-4-redesign-train-inference-split.md`. Splits
every backend into **train → infer → score** (trained model passed by reference, so an infer-side
failure re-uses the cached model and the AutoML table-name bug class cannot recur) feeding **one
shared score/track** step. **Phased:** Phase 1 = the split (custom, no new deps); Phase 2 = hybrid
GCPC serving + surviving 4A items. Each phase ends in a STOP checkpoint.

### Phase 1 — Train / Inference / Score split (custom; no new dependencies)

| # | Item | Status | Commit |
|---|------|--------|--------|
| P1.1 | `Forecaster` interface: `model_reference` + `attach_model` (base default; bqml/automl overrides) | done | 172a6b4 |
| P1.2 | `steps.py`: `train_backend_step` + `infer_backend_step`; shared `score_and_track_step` (model-config params); retire `run_backend_step` | done | 172a6b4 |
| P1.3 | `components.py`: `train_backend`/`infer_backend`/`score_and_track`; strip score+track from endpoint/batch predict | done | 172a6b4 |
| P1.4 | `pipeline.py`: rewire `_body` to train→infer→score per backend + TimesFM→score; `train:`/`infer:`/`score:` display names | done | 172a6b4 |
| P1.5 | Offline gate green + compile asserts train→infer→score edges; **cheap `--disable-automl` live run**; STOP | done | 172a6b4, 25a7400 |

Offline gate green (158 passed; ruff/format/ty clean) and the compiled DAG verified:
`build-tables → train:{m} → infer:{m} → score:{m} → compare`, TimesFM `register → deploy →
endpoint-predict → score:timesfm → compare`, teardown as ExitHandler. **Live `--disable-automl` run
`…-20260626114824` SUCCEEDED** end-to-end: BQML train→infer→score, TimesFM register→deploy→
endpoint-predict→score, compare winner **bqml_arima_xreg** (MAE 77.7 vs TimesFM 85.5), ExitHandler
teardown. AutoML deferred to the Phase 2 full run (avoids a throwaway ~2.5h; first live check of
`e8a0f5f`). **Phase 1 complete — STOP approved 2026-06-26.**

**Fan-in fix (25a7400):** `compare-backends` failed twice live before passing — KFP collects
`score_and_track` outputs into the `rows` list by *textual* substitution into the executor-input
JSON, so a dict output failed type binding and a raw-JSON-string output broke the JSON (unescaped
quotes). Fix: `score_and_track` returns **base64(JSON)**, `compare_backends` decodes; a contract test
asserts the output stays in the base64 alphabet so a regression fails offline, not in a live run.
Idiomatic artifact-based fan-in (replacing the base64 hop) is a Phase 2 cleanup candidate.

### Phase 2 — Hybrid GCPC serving + data prep + artifacts — COMPLETE

Folds in surviving 4A items. Ordered by user priority (lineage first) + dependency. GCPC pin verified
2026-06-26: `google-cloud-pipeline-components==2.22.0` declares `kfp<3,>=2.6.0` + `aiplatform<2,>=1.14.0`
→ our kfp 2.16.1 / aiplatform 1.158.0 both satisfy (real resolver dry-run still required before adding).

| # | Item | Status | Commit |
|---|------|--------|--------|
| P2.1 | **Data→serving lineage (4A.2+4A.3):** `build_tables` emits table-ref `Dataset` artifacts (`bq://` + config fingerprint + rows); wire into train/infer **and** TimesFM serving steps (closes edge-less serving branch) | done | be5383d |
| P2.2 | Self-bootstrapping data prep (4A.1): `ensure_source`/`ensure_prepped` front steps, existence + fingerprint guard | done | af65db6 |
| P2.3 | Hybrid GCPC serving (4A.6): `gcpc==2.22.0` after resolver dry-run; ModelUpload/EndpointCreate/ModelDeploy; AutoML+BQML stay custom | done | a8f06de |
| P2.4 | Richer artifacts (4A.4): `Output[Metrics]` on score, `Output[Markdown]` ranking on compare; *candidate:* replace base64 fan-in with artifact fan-in | done | 002812c |
| P2.5 | force_rebuild (4A.5) + machine right-size (4A.8, e2-standard-4→e2-standard-2) | done | ea9cad4 |
| P2.6 | Docs: hybrid-GCPC decision in CODE_STANDARDS + CLAUDE | done | 8374006 |
| P2.7a | **Cheap `--disable-automl` live run** — first live validation of P2.3 GCPC serving + P2.1/P2.2 data-prep+lineage + P2.4 artifacts + P2.5 sizing (BQML+TimesFM only) | done | c03b9ff, bf873d2, 9e8e581 |
| P2.7b | **One full AutoML run** = redesign acceptance + first live validation of `e8a0f5f` read fix + train/infer cache-reuse → Phase 2 STOP | done | 0df799c |

**P2.3 design notes (full-hybrid, approved 2026-06-26):** resolver dry-run clean (gcpc 2.22.0 adds only
itself; kfp 2.16.1 / aiplatform 1.158.0 satisfy). GCPC ops run in Google-managed images → no runtime
image rebuild; gcpc is a compile-time dep only. Endpoint path = `importer(UnmanagedContainerModel)` →
`ModelUploadOp` → `EndpointCreateOp` + `ModelDeployOp` (custom container ⇒ ModelUploadOp needs the
containerSpec via the importer, not `serving_container_*` args). **Deviations from the 4A.6 wording:**
(1) `ModelUndeployOp`/`EndpointDeleteOp` are **not** adopted — they need the in-handler model/endpoint
artifacts, which an `ExitHandler` exit task can't consume, so `teardown_serving` stays custom
(display-name based, runs on partial failure — the stranded-endpoint guard). (2) `endpoint_predict`
stays custom, now consuming the `VertexEndpoint` artifact's `resourceName` + ordered `.after(deploy)`.
(3) The `prepped` lineage edge moved from register→predict (the served model reads prepped at predict
time). (4) Batch path also uses `ModelUploadOp` (custom `batch_predict` consumes the `VertexModel`
artifact). **Correction (live-validated in P2.7a):** the `artifact_uri=""` baked-container assumption
was **wrong** — the GCPC `importer` rejects an empty URI (`INVALID_ARGUMENT`, "Failed to get the URI
of the artifact to import"). Fixed with `config.timesfm_artifact_uri` (a stable empty GCS prefix
`gs://…/serving/<model>/`; Vertex copies its empty contents to `AIP_STORAGE_URI`, the baked container
ignores it). See the P2.7a notes below.

**P2.5 design notes (2026-06-26):** Two independent sub-features. (1) **force_rebuild (4A.5):**
`DataConfig.force_rebuild: bool = False` + `submit.with_data_rebuild_forced` (model_dump→patch→
model_validate, no input mutation) + `--force-data-rebuild` CLI flag (standalone `if`, composes with
`--enable/--disable-automl`); the `ensure_source`/`ensure_prepped` components pass
`force=cfg.data.force_rebuild` into the already-`force`-aware steps. (2) **machine right-size (4A.8):**
`component_machine_type` was a DEAD config knob (declared + set in YAML, consumed nowhere) — so beyond
flipping the default `e2-standard-4`→`e2-standard-2` I **wired it** so the right-size is real. KFP
lightweight components expose no machine-type setter; on Vertex/GEAP Pipelines the CPU/memory limits
select the *closest-match* machine, so `config.component_resources` maps the machine type → `(cpu, mem)`
(`e2-standard-2`→`("2","8G")`, ×4/×8; raises `ValueError` on an unknown type — fail loud, no silent
fallback) and `pipeline._size` applies `set_cpu_limit`/`set_memory_limit` to the **custom** component
pods only. **GCPC ops (importer/model-upload/endpoint-create/model-deploy) are deliberately left
unsized** — they run in Google-managed images. Reviews-as-gates ([[sdd-reviews-as-gates]]): spec
reviewer ✅ + code-quality reviewer ✅ Approved (only minors; reviewer confirmed against GEAP docs that
cpu/mem limits → closest machine type). Offline gate 175 passed. Live machine-size effect first
observed in P2.7.

**P2.6 design notes (2026-06-26):** New authoritative section in `CODE_STANDARDS.md` ("Pipeline
components — GCPC vs custom (hybrid serving)") states the split rule for adding pipeline nodes: GCPC
ops **only** for opaque managed serving infra (importer → ModelUploadOp → EndpointCreateOp +
ModelDeployOp; standardized `google.Vertex*` artifacts + lineage; compile-time-only dep, no image
rebuild; test wiring via compile), custom `@dsl.component` shells for **all logic** (data prep,
build-tables, train/infer, shared score, compare, endpoint/batch predict; AutoML + BQML fully custom).
Records the three intentional deviations from a pure-GCPC lifecycle (custom display-name teardown via
ExitHandler since `ModelUndeployOp`/`EndpointDeleteOp` can't consume in-handler artifacts; custom
`endpoint_predict` consuming `VertexEndpoint.resourceName` `.after(deploy)`; `_size` applied to custom
pods only). `CLAUDE.md` gets a one-paragraph pointer to that section. Reviews-as-gates
([[sdd-reviews-as-gates]]): accuracy reviewer verified every claim against the code — all ACCURATE bar
one wording nit (kfp 2.16.1 is the *locked* version, constrained `>=2,<3` in pyproject, not
hard-pinned), fixed before commit. Docs-only; offline gate still green (175 passed).

**P2.7a design notes (2026-06-26):** First live run of the never-before-executed P2.3 GCPC serving
path, `--disable-automl` (BQML + TimesFM only) to skip the ~2.5h billable AutoML training. Required a
runtime image rebuild (components import `geaptimes` from the base image; digest `86a6e6dc…`). Two
live bugs found and fixed: (1) **importer `artifact_uri=""` rejected** — the P2.3 baked-container
assumption was wrong; fixed with `config.timesfm_artifact_uri` (empty GCS prefix) + a `.keep`
placeholder object. (2) **model-deploy 404 on a cached dead endpoint** — a cached `endpoint-create`
returned the prior run's endpoint that its ExitHandler teardown had already deleted; root cause is the
**KFP/Vertex caching serialization gotcha** (`set_caching_options(False)` → empty `cachingOptions` →
Vertex falls back to the pipeline-level default). Hot-fixed the run with pipeline-level
`enable_caching=False`; the durable fix inverts the design (`9e8e581`): pipeline-level always off,
producers opt back in per-task with explicit `True`, plus a `--no-cache` flag. **Run
`…-20260626155800` SUCCEEDED** end-to-end: winner `bqml_arima_xreg` (MAE 77.72 / RMSE 101.72) vs
`timesfm` (85.51 / 113.65) — matches the Stage 3 baseline (behavior-preserving redesign); transient
endpoint torn down (no strand); P2.4 Markdown ranking + comparison JSON correct. Confirms there is **no
serving-container bug** — both earlier deploy failures were the cached-dead-endpoint 404. Reviews-as-
gates ([[sdd-reviews-as-gates]]): caching-redesign reviewer flagged the importer `--no-cache`
consistency gap (importer defaults to cacheable), fixed before the gate went green (181 passed). Full
live-run writeup in `docs/notes/stage-4-managed-pipelines-live-run.md`.

**P2.7b design notes (2026-06-26):** The full three-backend run (`--no-cache`, clean from-scratch),
run `…-20260626174237`, **SUCCEEDED** end-to-end — the Phase 2 / redesign acceptance. AutoML's
`train-backend → infer-backend` chain ran green: **first live validation of the `e8a0f5f` read fix**
(the read-side bug that killed the original ~2.5h run after training) and proof the train/infer split
works (the trained model is passed as an artifact, so an infer-side failure would re-use it instead of
re-training). Acceptance (9/9, see `/tmp/geaptimes_p27b_verify.py`): pipeline + all tasks
SUCCEEDED/SKIPPED; **3 ExperimentRuns** under `citibike-daily-baseline` (one per backend); winner
**`bqml_arima_xreg`** MAE 77.72 / RMSE 101.72, then `timesfm` 85.51 / 113.65, then `automl` 142.46 /
182.83; transient endpoint **and** model torn down (full self-cleanup — `teardown_serving_step`
deletes both when `keep_deployed=false`); `solution=geaptimes` label on the pipeline job. AutoML scored
worst — a **modeling** outcome (1000 milli-node-hour budget), not a pipeline issue; the point of P2.7b
was proving the full three-backend path including AutoML, which it did. Winner matches the P2.7a
baseline → the redesign is behavior-preserving. (An ADC token lapse mid-run killed the background
pollers but not the run; re-auth + re-poll showed SUCCEEDED.) **Phase 2 complete — STOP for approval.**

**P2.4 design notes (2026-06-26):** `score_and_track` now emits `Output[Metrics]` (per-backend MAE /
RMSE scalars, via `metrics.log_metric`) shown on the task in the Vertex Pipelines UI — in addition to
the Vertex `ExperimentRun` logged inside the step. `compare_backends` now emits `Output[Markdown]`: a
ranking table (rank / model / mae / rmse, winner row flagged, lowest RMSE first) alongside the
existing comparison `Dataset` JSON + winner return. Both are additive — `pipeline.py` is unchanged
(KFP auto-creates the output artifacts; callers don't pass them). **Base64 fan-in candidate: kept, not
replaced.** Idiomatic artifact-based fan-in needs `dsl.Collected`, which only works under
`dsl.ParallelFor` — and ParallelFor is deferred (see `docs/notes/pipeline-parallelfor-deferred.md`),
since the backends are a heterogeneous compile-time Python loop, not a uniform runtime fan-out. For a
static list of producer tasks, KFP can only fan in via a `list` *parameter* (built by textual
substitution), so the base64 encoding remains the correct JSON-/substitution-safe hop; a contract test
keeps it from regressing.

---

## Tier 2 — Stage 3 (Experiment Tracking & DOE Framework) — COMPLETE

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
| 3.8 | Live run on `hybrid-vertex` (TimesFM + BQML) + notes | done | `92b6ac1` |

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
