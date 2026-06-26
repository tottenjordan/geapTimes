# Stage 4 — managed pipelines live run (P2.7a)

First live execution of the redesigned hybrid GCPC comparison pipeline on Vertex AI. This is the
`--disable-automl` validation (P2.7a): it proves the full Phase 2 path **except** AutoML (BQML +
TimesFM only). The one remaining full-AutoML run is P2.7b (the Phase 2 STOP).

## Environment

- Project `hybrid-vertex` (934903580331), region `us-central1`, bucket `geaptimes-hybrid-vertex`,
  BQ dataset `geaptimes`, experiment `citibike-daily-baseline`.
- Runtime image `us-central1-docker.pkg.dev/hybrid-vertex/geaptimes/geaptimes-runtime:latest`,
  digest `sha256:86a6e6dcc1dd1ef05dc6e7cf02155a47e2d51968ce9b317c3f7e40cd30afbb64` (Cloud Build
  `458c07d0`). **A rebuild was required**: KFP lightweight components `COPY src` + `pip install .`,
  so they import `geaptimes` from the base image — any change to an imported module needs a new image.
- Submit: `GCP_PROJECT=hybrid-vertex uv run python -m geaptimes.pipelines.submit --config
  config/comparison_config.yaml --disable-automl` (config uses `${GCP_PROJECT}` substitution).

## Result

Run `geaptimes-comparison-top25-h14-t14-v14-20260626155800` **SUCCEEDED** end-to-end, including the
never-before-reached serving chain `importer → model-upload → endpoint-create → model-deploy →
endpoint-predict → score:timesfm → compare-backends → teardown-serving`.

- Winner **`bqml_arima_xreg`** — MAE 77.72 / RMSE 101.72 — vs `timesfm` MAE 85.51 / RMSE 113.65.
  Matches the Stage 3 baseline, confirming the train/infer split + hybrid redesign is
  behaviour-preserving.
- P2.4 `ranking_md` Markdown renders with the winner flagged; comparison `Dataset` JSON correct.
- Transient endpoint **torn down** (no stranded endpoint). There is **no serving-container bug** —
  both earlier `model-deploy` failures were 100% the cached-dead-endpoint 404 (below).

This validates live: P2.1/P2.2 self-bootstrapping data prep + lineage, P2.3 hybrid GCPC serving,
P2.4 richer artifacts, P2.5 machine right-size. Only AutoML (P2.7b) remains.

## Two live bugs in the never-before-run P2.3 GCPC serving path

### 1. importer rejects an empty `artifact_uri` (`c03b9ff`)

The P2.3 assumption that a baked-checkpoint custom container could pass `artifact_uri=""` was
**wrong**. The GCPC `importer(UnmanagedContainerModel)` fails `INVALID_ARGUMENT` ("Failed to get the
URI of the artifact to import") on an empty URI. (Vertex `Model.upload` allows an empty `artifact_uri`
at REST for custom containers, but the KFP importer node always needs a real GCS URI.)

Fix: `config.timesfm_artifact_uri(cfg)` → `gs://{bucket}/serving/{timesfm_model_display_name}/`, a
stable **empty** GCS prefix. Vertex copies its (empty) contents to `AIP_STORAGE_URI` at deploy; the
baked container ignores it (`HF_HUB_OFFLINE=1`). A `.keep` placeholder object holds the prefix.

### 2. ephemeral endpoint lifecycle was cached across runs → deploy 404

`model-deploy` failed with `Endpoint .../<id> is not found`. Cause: a cached `endpoint-create` handed
back the **previous** run's endpoint, which that run's `ExitHandler` teardown had already deleted, so
the deploy targeted a 404. ("replica exited status 13" is just the GCPC deploy launcher exiting after
the 404 — not a container crash.)

The durable fix is the caching inversion below.

## Vertex caching gotcha (the durable fix — `9e8e581`)

**`task.set_caching_options(False)` is ineffective on Vertex.** KFP serializes disabled caching as an
**empty** `cachingOptions={}` (proto-JSON omits false booleans), so Vertex can't tell "explicitly
false" from "unset" and falls back to the **pipeline-level** `enable_caching` default. With that
default `True`, the lifecycle tasks that "set False" were actually cached. Only
`set_caching_options(True)` survives serialization (renders `{enableCache: true}`) and is honored.

The first hot-fix (`bf873d2`, per-task `set_caching_options(False)` on the lifecycle) was therefore
**ineffective**; the green run only happened once submitted with a pipeline-level `enable_caching=False`.

Durable design (inverted, `9e8e581`):

- `submit_pipeline` **always** submits pipeline-level `enable_caching=False` (the off baseline). The
  misleading `enable_caching` submit param was dropped.
- Deterministic **producers** opt back in per-task via `set_caching_options(caching)` where
  `caching = cfg.pipeline.enable_caching` (default `True`). Explicit `True` survives serialization and
  caches. The `importer` is included — it defaults to cacheable, so it must honor the flag explicitly
  or it would stay cached under `--no-cache`.
- The ephemeral **lifecycle** (`endpoint-create`, `model-deploy`, `endpoint-predict`/`batch-predict`,
  `teardown-serving`) stays `set_caching_options(False)` → empty → falls back to the off default →
  never caches.
- New `--no-cache` flag (`submit.with_caching_disabled`) sets `pipeline.enable_caching=False` for a
  full from-scratch re-run.

Lesson: never let a task that manages or consumes a transient resource (which another task destroys)
be cacheable; and remember that on Vertex "disabled caching" means "inherit the pipeline default", not
"off".

## Run log

| Run suffix | HEAD | Outcome |
|---|---|---|
| `…150639` | pre-fix | FAILED — importer empty `artifact_uri` (bug 1) |
| `…152524` | `c03b9ff` | FAILED — model-deploy 404, cached dead endpoint (bug 2) |
| `…154754` | `bf873d2` | FAILED — same 404 (per-task `False` ineffective on Vertex) |
| `…155800` | `bf873d2` + inline `enable_caching=False` | **SUCCEEDED** — full chain green |

Polling: `aiplatform.PipelineJob.get(RN)` — pipeline state 3=RUNNING/4=SUCCEEDED/5=FAILED; task state
3=SUCCEEDED/7=FAILED/8=SKIPPED(cached)/9=NOT_TRIGGERED.

Inspecting serving-container logs: filter `resource.type="aiplatform.googleapis.com/Endpoint"` **by
endpoint_id** — the shared `hybrid-vertex` project has many other endpoints (taxifare, churn,
vertex-custom-serve) whose `/health` logs are noise.
