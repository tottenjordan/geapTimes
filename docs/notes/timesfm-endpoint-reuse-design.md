# TimesFM endpoint reuse — design & effort estimate

Captured 2026-06-26 from Stage 4 pipeline feedback (item #6). **Status: deferred to Stage 5**
(coupled to the warm/leave-running endpoint feature, which is already Stage 5 scope). This note
records the design so it can be picked up directly when warm endpoints land.

## The ask

Before registering + deploying a fresh TimesFM model each run, check whether a *similar* deployment
already exists and, if so, reuse it for the endpoint-predict step instead of deploying a new one.

## Why it's not a drop-in today

The comparison DAG's default cost posture is **transient** (`pipeline.serving.keep_deployed = False`):
`deploy_endpoint_step` creates an endpoint + deploys, and the `dsl.ExitHandler` runs
`teardown_serving_step`, which **undeploys + deletes by display name** at the end of every run. So in
the default mode there is nothing left to reuse next time — reuse only makes sense alongside a
**warm/persistent endpoint** (`keep_deployed = True`), and "leave-running endpoint as a default" is
listed as Stage 5 out-of-scope in the approved Stage 4 plan.

Also note: **same-config re-runs already get reuse for free** via KFP cross-run caching — when
`config_json` is unchanged (within the cache window), the `register-timesfm` / `deploy-endpoint`
tasks are SKIPPED. The new value of explicit reuse is *across* different pipeline runs / configs that
share the same serving fingerprint.

## Proposed design (for the warm-endpoint world)

1. **Deployment fingerprint.** A TimesFM deployment is interchangeable iff its serving contract
   matches: the resolved image **digest** (the checkpoint is baked into the image) plus the serving
   env vars (`TIMESFM_CHECKPOINT`, `TIMESFM_MAX_CONTEXT`, `TIMESFM_CONTEXT_LEN`, `TIMESFM_HORIZON`,
   `TIMESFM_QUANTILES`). Compute `fingerprint = sha256(image_digest + serving_env)` and stamp it as a
   label on the deployed model / endpoint (alongside `solution=geaptimes`).
2. **Lookup step.** A `find_reusable_endpoint_step(cfg)` lists endpoints by
   `labels.fingerprint=<fp>` (and `labels.solution=geaptimes`) and returns the resource name of a
   healthy match (a deployed, in-service model) or `""`.
3. **DAG branch.** If a match is found: skip `register-timesfm` + `deploy-endpoint`, feed the found
   endpoint resource name straight into `endpoint-predict`. Otherwise: the current
   register → deploy → predict path.
4. **Teardown guard.** Teardown must only remove what *this run created* — never a reused warm
   endpoint. Track an "owned/created-this-run" flag (or skip teardown entirely when a reused endpoint
   was used) so the ExitHandler doesn't delete a shared warm endpoint.

## Challenges

- **Fingerprint on digest, not the `:latest` tag.** The image is published as `:latest`; the digest
  behind it changes on rebuild. Fingerprinting on the tag would risk serving a stale model — resolve
  the digest (an Artifact Registry describe call) before hashing.
- **Teardown semantics.** Current teardown deletes by display name unconditionally; needs the
  "only tear down what I created" guard above.
- **Concurrency / races.** Two runs that share a display name can race to create/reuse the same
  endpoint; reuse-by-fingerprint-label is safer than reuse-by-display-name but still needs a
  create-if-absent that tolerates a concurrent winner.

## Effort estimate

~0.5–1 day of code (a `find_reusable_endpoint_step` + a fingerprint label on deploy + a DAG branch +
the teardown guard + offline tests) **plus one live re-run** to verify. Only worthwhile once warm
endpoints are a supported mode — until then KFP caching covers the realistic same-config case.

## Recommendation

Defer to Stage 5, implemented together with the warm/leave-running endpoint feature. Adopt the
**fingerprint-label** approach as the reuse key (not display-name matching).
