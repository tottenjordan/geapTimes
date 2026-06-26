# Code Standards

Authoritative standards for **geapTimes**. All code and environment changes must adhere to
this document. (`CLAUDE.md` links here.)

## Python

- **Version:** Python **3.11** (`requires-python >=3.11`, pinned via `.python-version`).
- **Package/dependency management:** **`uv` for everything**. Never use bare `pip` or `python`.
  - `pyproject.toml` (+ committed `uv.lock`) is the **source of truth** for dependencies.
  - Add/remove deps with `uv add` / `uv remove` — do **not** hand-edit the `dependencies` or
    `dependency-groups` sections.
  - Run commands with `uv run ...` (never manually activate a venv).
  - `requirements.txt`, when needed for pipeline/custom containers, is **generated**:
    `uv export --no-hashes --no-dev -o requirements.txt`. It is a build artifact, not a source.

## Linting, formatting, typing, testing

- **Lint + format:** **`ruff`** (never black / flake8 / isort). `uv run ruff check .` and
  `uv run ruff format .`.
- **Type checking:** **`ty`** (Astral's type checker). `uv run ty check`.
- **Tests:** **`pytest`**. `uv run pytest`.
- Config for all of the above lives in `pyproject.toml`.

## Cloud preflight (validate before deploying a pipeline)

The offline gate (`ruff check` → `ruff format --check` → `ty check` → `pytest`) is mandatory before
any commit, but it cannot see Vertex's **server-side** validation of managed jobs (AutoML, pipeline
components). Reaching that validation through a full pipeline deploy costs a container rebuild +
submit + pod startup (~5 min per iteration). Catch those failures cheaply instead:

- **Encode every server-side rule you discover as an offline invariant** with a unit test, so a
  regression fails at `pytest` time for free. Example: `AutoMLForecaster.validate_config()` checks
  the Vertex AutoML Forecasting column-role rules (target ∈ unavailable-at-forecast, time column ∈
  available-at-forecast, the series identifier carries **no** transformation, every feature/target
  carries a transformation, quantiles ⇒ `minimize-quantile-loss`).
- **Run the live preflight before submitting** to exercise Vertex's real `create_training_pipeline`
  validation in seconds, with **no training spend**:
  ```bash
  GCP_PROJECT=hybrid-vertex uv run python -m geaptimes.pipelines.submit \
      --config config/comparison_config.yaml --preflight-automl
  ```
  It runs `fit(sync=False)` and waits only for the training-pipeline *resource* to be created (where
  column/role validation runs), then cancels + deletes the probe job (it reaches `CANCELLED` before
  any billable compute). A clean `AutoML preflight PASSED` means the config will clear validation in
  the deployed pipeline.
- Every new server-side error must be fixed **and** added to the static validator + a test, so the
  slow cloud loop is paid at most once per rule.
- **Iterate on the pipeline without the AutoML cost.** AutoML training dominates the comparison
  run's wall-clock (~1–2 h) and is the only billable-compute backend, while every other task
  finishes in minutes. When testing pipeline mechanics (a new image, artifact wiring, GCPC
  components), submit with `--disable-automl` to skip it; submit the full three-backend comparison
  (`--enable-automl`, or a config with AutoML enabled) only when you specifically need AutoML
  results:
  ```bash
  GCP_PROJECT=hybrid-vertex uv run python -m geaptimes.pipelines.submit \
      --config config/comparison_config.yaml --disable-automl
  ```

## Source layout

- `src/` layout: the importable package is `src/geaptimes/`. Import as `from geaptimes... import ...`.
- Config in `config/`, notebooks in `data_notebooks/`, CLI/ops scripts in `scripts/`,
  durable session notes in `docs/notes/`, archived plans in `docs/plans/`.

## Planning artifacts

- `PLANS.md` is the **living tracker** (Tier 1 roadmap + Tier 2 items with status and commit
  hashes); edited in place, history in git.
- Each **approved per-stage plan** is snapshotted to `docs/plans/NNN_<slug>.md` at approval time
  (immutable; `<slug>` is the plan title lowercased with spaces → `-`). See
  [`docs/plans/README.md`](./docs/plans/README.md). Do not edit a snapshot after the fact —
  course corrections live in `PLANS.md` and git.

## Configuration

- Experiments are defined in **YAML** and parsed into **Pydantic v2** models
  (`src/geaptimes/schemas.py`) for type safety. Code consumes the validated `ExperimentConfig`,
  never raw YAML.
- The GCP project is supplied via the `GCP_PROJECT` env var (default `hybrid-vertex`); see
  `.env.example`. Do not hardcode project ids in committed config.

### Table naming

- Derive all output table names from `geaptimes.naming` (never hardcode). Convention
  (human-readable tokens): the shared `source` table uses the data slug (`__top25`);
  `prepped`/`train`/`infer` use the full config slug (`__top25_h14_t14_v14`).
- Token slugs do not encode covariate/weather/holiday changes — disambiguate experiments via a
  distinct `execution.experiment_name`. Each built table's **description** carries the full config
  fingerprint (`geaptimes.naming.config_fingerprint`) for traceability.

## Notebooks

- Use the standardized **`geaptimes`** Jupyter kernel (backed by the `uv` venv) so every user
  runs the same environment. Register once per machine:
  `uv run python -m ipykernel install --user --name geaptimes --display-name "geapTimes (uv 3.11)"`.
- Notebooks pin `metadata.kernelspec.name = "geaptimes"`.
- Notebooks are excluded from `ruff` (`extend-exclude`); keep heavy logic in the package and
  import it, so it is linted/typed/tested.
- **Colab option (not default):** to run a notebook in Colab, add a guarded first cell that
  installs the package (`pip install -e .` from a clone, or from a release) and authenticates via
  `google.colab.auth.authenticate_user()`. Keep this opt-in so the local `uv` flow stays primary.

## GCP resource labels

- **Every label-capable GCP asset must carry the label `solution=geaptimes`.** This includes BQ
  datasets and tables, GCS buckets, and (going forward) BQ query jobs, Vertex AI experiments,
  pipelines, models, and endpoints.
- Use the single source of truth: `geaptimes.constants.RESOURCE_LABELS` (and
  `geaptimes.constants.bq_labels_option()` for BigQuery DDL). Do not hardcode the label inline.
- New scripts/code that create GCP resources must apply these labels at creation time.

## Git / commits / PRs

- **Never** add `Co-Authored-By` trailers to commits, and never add a "Generated with …" line
  to PR bodies.
- Work on a branch off the default branch unless told otherwise; commit/push only when asked.

## Pre-commit

- Hooks are defined in `.pre-commit-config.yaml` but are **not active** until installed:
  `uv run pre-commit install`. Run on demand with `uv run pre-commit run --all-files`.

## Deferred / backlog (not yet adopted)

- Security pre-commit hooks (detect-secrets, shellcheck, actionlint, zizmor) from the
  `modern-python` skill are deferred; revisit before any public release.
- **Optional quantile forecasting (Stage 5).** Quantiles are currently always emitted
  (`QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]`), which forces AutoML's `optimization_objective` to
  `minimize-quantile-loss` and bakes the `qXX` columns into every prediction frame. Make this a
  config switch: when quantiles are enabled, keep the current behaviour (force
  `minimize-quantile-loss`, emit `qXX`, and add **quantile loss** to the metric suite); when
  disabled, drop the `qXX` columns, relax the AutoML objective constraint (allow e.g.
  `minimize-rmse`), and skip quantile metrics. Touches `models/base.py` (column constants),
  `schemas.py` (a `quantiles` flag at the data/forecast level), `models/automl.py`
  (`validate_config()` + `run_kwargs`), the metric suite, and the serving env (`TIMESFM_QUANTILES`).
