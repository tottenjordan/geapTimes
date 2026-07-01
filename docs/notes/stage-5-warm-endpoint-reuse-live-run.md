# Stage 5 — warm-endpoint reuse live run (5.5)

Live validation of the 5.4 warm-endpoint reuse feature: a `keep_deployed=True` + `reuse_endpoint=True`
run deploys a TimesFM endpoint once, a second run **reuses** it (no redeploy), and the transient
teardown's guard refuses to delete it. Run `--disable-automl` (BQML + TimesFM only) to keep it cheap.

## Environment

- Project `hybrid-vertex` (934903580331), region `us-central1`, bucket `geaptimes-hybrid-vertex`,
  BQ dataset `geaptimes`, experiment `citibike-daily-baseline`.
- Runtime image `us-central1-docker.pkg.dev/hybrid-vertex/geaptimes/geaptimes-runtime:latest`,
  digest `sha256:cc3a5c0044d5512425bde6d20316451cc38d17f0290f2045e0788b6d88919bd8`
  (Cloud Build `80355d6b`, rebuilt for 5.4 — see the stale-image gotcha below).
- Config: `config/comparison_config.yaml` with `--disable-automl` and serving overrides
  `keep_deployed=true` + `reuse_endpoint=true` (applied in-process; reuse is a config switch, not a
  CLI flag).

## Stale-image gotcha (root cause of the first failed run)

The first warm-deploy submit **failed** at `ensure-source` with:

```
pydantic_core ValidationError for ExperimentConfig
pipeline.serving.reuse_endpoint
  Extra inputs are not permitted [type=extra_forbidden, input_value=True]
```

KFP lightweight components `COPY src` + `pip install .`, so they import `geaptimes` from the **baked
image**, and the image predated 5.4. The submitted `config_json` carried the new
`serving.reuse_endpoint` field, but the old code's schema (`_Base` is `extra="forbid"`) rejected it.

**Lesson (corollary to the Stage-4 rebuild rule):** adding a *schema field* — not just changing an
imported function — requires rebuilding the runtime image before any live pipeline run, because every
custom component deserializes `config_json` with the baked schema. Fix: `gcloud builds submit --config
src/geaptimes/pipelines/container/cloudbuild.yaml …` (digest changed `86a6e6dc…` → `cc3a5c00…`; the
serving fingerprint changed with it, by design — the fingerprint hashes the digest).

The failed run also stranded an **empty** endpoint + a registered model (it was cancelled with
`endpoint-create` done and `model-deploy` never reached). A `keep_deployed` run emits **no** teardown
task, so nothing auto-cleans a cancelled warm run — these were deleted explicitly by resource name.
(An endpoint with zero deployed models runs no machine, so there was no serving compute cost.)

## Sequence + results

1. **Warm deploy** — `geaptimes-comparison-top25-h14-t14-v14-20260630232242` **SUCCEEDED**. Fresh
   serving chain `importer → model-upload → endpoint-create → model-deploy → endpoint-predict`, and
   **no `teardown-serving` node** (`keep_deployed` ⇒ `needs_teardown=False`). Left one warm endpoint
   `endpoints/4452996253969547264` + model `models/5287191327916687360`, both labelled
   `fingerprint=82717760d4b5126b`, `keep=true`, `solution=geaptimes`.
2. **Reuse** — `geaptimes-comparison-top25-h14-t14-v14-20260701024425` **SUCCEEDED**. Submit-time
   lookup logged `reusing warm TimesFM endpoint …/4452996253969547264 (fingerprint 82717760…)`; the
   compiled DAG contained **`endpoint-predict-by-name`** and **omitted** the entire serving lifecycle
   (`importer`/`model-upload`/`endpoint-create`/`model-deploy`) **and** `teardown-serving`. No
   redeploy — it predicted straight against the warm endpoint.
3. **Teardown guard (live probe)** — ran `teardown_serving_step` against the real project with the
   warm endpoint present. It logged `teardown guard: skip kept-warm endpoint …/4452996253969547264`
   and `… skip kept-warm model …/5287191327916687360`; both resources survived untouched
   (`deployed=1`). This is the destructive-regression guard: a transient run that shares the
   config-derived display name will not delete a shared warm deployment.
4. **Cleanup** — warm endpoint undeployed + deleted and model deleted explicitly (bypassing the
   guard, by resource name). Post-cleanup: 0 TimesFM endpoints, 0 TimesFM models.

## Metrics (richer suite validated live)

Both runs recorded the **full 5.1/5.2 eval suite** to the experiment — the live proof that sMAPE +
quantile loss now flow through the pipeline scorer (older 06-26 runs had these as `NaN`). n=308.

| Run        | Backend           | MAE    | RMSE    | sMAPE  | Quantile loss |
| ---------- | ----------------- | ------ | ------- | ------ | ------------- |
| Warm deploy | timesfm          | 85.51  | 113.65  | 39.26  | 31.30         |
| Warm deploy | bqml_arima_xreg  | 77.72  | 101.72  | 42.66  | 30.27         |
| Reuse       | timesfm          | 85.51  | 113.65  | 39.26  | 31.30         |
| Reuse       | bqml_arima_xreg  | 77.72  | 101.72  | 42.66  | 30.27         |

- **timesfm is identical across the two runs** — the reuse path predicts against the same warm
  endpoint and reproduces the result exactly (reuse correctness).
- Ranking by RMSE: `bqml_arima_xreg` (101.72) beats `timesfm` (113.65) — matches the Stage 3/4
  baseline, so the reuse machinery is behaviour-preserving.

## Takeaways

- Warm-endpoint reuse works end-to-end: fingerprint-labelled warm deploy → submit-time discovery →
  compile-time reuse branch → correct predictions with no redeploy.
- The teardown guard protects a shared warm endpoint from a transient run (validated live).
- Operational rule: a warm (`keep_deployed`) run leaves resources that no pipeline teardown reclaims;
  clean them up explicitly by resource name (the guard blocks `teardown_serving_step` from doing so).
- Any schema change requires a runtime-image rebuild before a live run.
