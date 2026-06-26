# geapTimes — Stage 4 Redesign: Train/Inference Split + Hybrid GCPC (Phased)

> Approved 2026-06-26. Immutable snapshot (course corrections live in PLANS.md + git).
> Supersedes the "Stage 4 Addendum" (4A.1–4A.8); those items are folded into the phases below.

## Context

The first live comparison run (`…-20260626063508`) **failed** in the AutoML backend: `_read_output_table`
hardcoded the predictions table name, 404'ing after ~2.5h of successful training. The bug is fixed
(`e8a0f5f`), but it exposed a structural flaw — the `run_backend` component welds **fit + predict +
score + track** into one pod, so a trivial read-side bug threw away the entire training run, and there
is no model artifact to re-use.

Acting on three pieces of user feedback (discussed 2026-06-26):
1. **Minimize tasks per component** — give each backend dedicated **Train** and **Inference** steps,
   feeding **one shared score/track** step.
2. **Reconsider GCPC vs custom SDK-wrapper components** — decision: **Hybrid** (GCPC for the TimesFM
   serving lifecycle only; custom SDK-wrapper components for AutoML train/predict; BQML stays custom).
3. **`dsl.ParallelFor`** — **deferred**; backends already run in parallel and their chains are
   heterogeneous (see `docs/notes/pipeline-parallelfor-deferred.md`).

**Intended outcome:** a comparison DAG where each backend is `train → infer → score`, with the trained
model passed as an artifact between train and infer — so an inference-side failure re-uses the cached
trained model (no 2.5h re-train) and the AutoML table-name bug class cannot recur.

**Rollout = Phased**, each phase ending in a STOP checkpoint:
- **Phase 1** — the split (custom only, no new deps); validate offline + one cheap `--disable-automl` run.
- **Phase 2** — hybrid GCPC serving + self-bootstrapping data prep + richer artifacts + machine
  right-size; validate offline + **one full AutoML run** (also the first live validation of the
  `e8a0f5f` read fix and the train/infer cache-reuse win).

---

## Phase 1 — Train / Inference / Score split (custom; no new dependencies)

### Design

```
build-tables
  ├─ (per in-process model)  train-backend → infer-backend → score-and-track ─┐
  └─ timesfm: register → deploy → endpoint-predict ──────────→ score-and-track ┤→ compare-backends
                     (or)        → batch-predict   ──────────→ score-and-track ┘
  [ExitHandler: teardown-serving]  (unchanged)
```

- **Train and Infer are generic** over the `Forecaster` interface (heterogeneity lives *inside* the
  forecaster, not in the DAG).
- **TimesFM stays a bespoke chain** (no training; register/deploy/serve are genuinely different).
- **score/track is the one shared component** — already exists as `score_and_track_step`
  (`steps.py:258`); becomes the single tracking point for all backends.

### `Forecaster` interface (`models/base.py`) — two small additions

- `model_reference` → `str`: the trained-model handle. Default `""`.
- `attach_model(reference: str)` → `None`: prepare `predict()` to use an already-trained model
  without re-`fit()`. Default no-op.

Overrides:
- `BQMLForecaster` (`models/bqml.py`): `model_reference` returns `self.model_id` (config-derived);
  `attach_model` is a no-op (`predict()` already references the model id via `geaptimes.naming`).
- `AutoMLForecaster` (`models/automl.py`): `model_reference` returns `self.model.resource_name`
  after `fit()`; `attach_model(ref)` sets `self.model = self._resolve_aiplatform().Model(ref)` so
  `predict()` runs against the already-trained model. **This makes a failed infer cheap to retry.**
- `TimesFMForecaster`: unaffected (not routed through train/infer).

### `steps.py`

- **Add** `train_backend_step(cfg, model_name, *, forecaster=None) -> str` — build forecaster, `fit()`,
  return `model_reference`.
- **Add** `infer_backend_step(cfg, model_name, model_reference, *, forecaster=None) -> pd.DataFrame`
  — build forecaster, `attach_model(reference)`, `predict()`, return the standardized frame.
- **Keep** `score_and_track_step` (`:258`) as the shared scorer; extend it to `log_params` the model
  config so the ExperimentRun is complete for every backend.
- **Retire** `run_backend_step` (`:196`) — pipeline-only; tracking moves into `score_and_track_step`.

### `components.py`

- **Add** `train_backend(config_json, model_name, tables) -> str` and
  `infer_backend(config_json, model_name, model_reference, predictions: Output[Dataset])`.
- **Add** shared `score_and_track(config_json, model_name, predictions: Input[Dataset]) -> dict`.
- **Strip** score/track out of `endpoint_predict` and `batch_predict` (predictions `Dataset` only).
- `build_tables`, `register_timesfm`, `deploy_endpoint`, `teardown_serving`, `compare_backends` unchanged.

### `pipeline.py` (`_body`)

- Per in-process model: `train_backend(tables=tables.output)` → `infer_backend(model_reference=tr.output)`
  → `score_and_track(predictions=inf.outputs["predictions"])` → `rows.append(sc.output)`.
- TimesFM: `endpoint_predict`/`batch_predict` (predictions-only) → `score_and_track` → `rows.append`.
- Display names: `train:{model}`, `infer:{model}`, `score:{model}`, `timesfm:endpoint-predict` /
  `timesfm:batch-predict`, `score:timesfm`. ExitHandler/teardown unchanged.

### Tests

- `test_pipeline_steps.py` — `train_backend_step` / `infer_backend_step` (fake forecaster recording
  `fit`/`attach_model`/`predict`); keep `score_and_track_step`.
- `test_automl.py` / `test_bqml.py` — `attach_model` + `model_reference`.
- `test_pipeline_compile.py` — new executor/edge/display-name assertions (`exec-train-backend`,
  `exec-infer-backend`, `exec-score-and-track`).
- `test_pipeline_submit.py` — update `_run_backend_executors` names.

### Phase 1 verification

- Offline: `ruff check` → `ruff format --check` → `ty check` → `pytest`.
- `compile_pipeline(comparison_config)` emits the new DAG (assert train→infer→score edges).
- **Cheap live run**: rebuild image, submit `--disable-automl` → confirm BQML `train→infer→score`,
  TimesFM `register→deploy→endpoint-predict→score`, ExitHandler teardown, comparison artifact.
- **STOP checkpoint.**

---

## Phase 2 — Hybrid GCPC serving + self-bootstrapping data prep + artifacts (outline)

- **Hybrid GCPC serving** (4A.6 revised): `uv add google-cloud-pipeline-components` (pin compatible with
  aiplatform 1.158.0 / kfp 2.16.x — verify first). Replace custom `register_timesfm`/`deploy_endpoint`/
  `teardown_serving` with `ModelUploadOp`/`EndpointCreateOp`+`ModelDeployOp`/`ModelUndeployOp`+
  `EndpointDeleteOp`. Keep `endpoint_predict` custom. AutoML + BQML stay custom SDK-wrappers
  (`AutoMLForecastingTrainingJobRunOp` stays out of scope).
- **Self-bootstrapping data prep** (4A.1–4A.3): `ensure_source` → `ensure_prepped` with existence +
  `config_fingerprint` guard; table-reference artifacts; wire `prepped` into build_tables + TimesFM serving.
- **Richer artifacts** (4A.4): `Output[Metrics]` on infer/score, `Output[Markdown]` on compare.
- **Freshness override** (4A.5): `data.force_rebuild` + `--force-data-rebuild`.
- **Right-size component machine** (4A.8): `component_machine_type` `e2-standard-4` → `e2-standard-2`.
- **Docs**: hybrid GCPC decision in `CODE_STANDARDS.md` + `CLAUDE.md`.
- **Full live run** = redesign acceptance (3 ExperimentRuns, AutoML train→infer cache-reuse, transient
  teardown, comparison artifact + winner, all labelled `solution=geaptimes`; first live validation of
  `e8a0f5f`). **STOP** + `docs/notes/stage-4-managed-pipelines-live-run.md` + PLANS finalize.

## Risks

- GCPC version compatibility (Phase 2) — verify the pin before adopting; else keep serving custom.
- AutoML `attach_model` must resolve a `Model` by resource name without re-`fit()` — unit-tested;
  first proven live in the Phase 2 full run.
- One full 2.5h AutoML run total (Phase 2); Phase 1 stays cheap via `--disable-automl`.
