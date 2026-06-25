# geapTimes — Stage 1 Plan (with Master Roadmap)

> Archived snapshot of the approved Stage 1 plan (reconstructed). Immutable — see
> [`PLANS.md`](../../PLANS.md) and git for what actually shipped.

## Context
We are refactoring an older, notebook-heavy, M5-based repository
(`tottenjordan/vertex-forecas-repo`) into a modern, production-grade, **config-driven**
forecasting framework — **geapTimes** — on the Gemini Enterprise Agent Platform (Vertex AI;
SDK `google-cloud-aiplatform`). The build runs in 5 staged milestones, each ending in a hard
**STOP** checkpoint for explicit approval.

This plan details **Stage 1 — Data Architecture & Config Schema** and keeps Stages 2–5 as a
coarse roadmap (expanded just-in-time when reached). Stage 1 establishes: the data layer (daily
NYC Citibike departures per station, top-25, joined with NOAA GSOD weather + station metadata,
mirroring statmike's BQ prep pattern), the type-safe YAML→Pydantic config (the contract later
stages consume), and GCP provisioning.

## Locked decisions (from discussion)
- **Packaging:** `pyproject.toml` managed by `uv` is source of truth; `requirements.txt` generated via `uv export` for containers. **Python 3.11.**
- **Layout:** src-layout package `src/geaptimes/` (deviates from the blueprint's flat `src/`).
- **Tooling:** `ruff` (lint+format), `pytest`, **`ty`** (Astral's type checker, per the `modern-python` skill). No `Co-Authored-By` / "Generated with" trailers.
- **Target:** `num_trips = COUNT(*) GROUP BY start_station_name, DATE(starttime)`; series = `start_station_name`; top-25 stations.
- **Splits:** statmike pattern — `_source` → `_prepped` with a `splits` column by date range off global max (`TEST=14`, `VALIDATE=14`); horizon `14`; gap-fill within each station's active window.
- **Covariates (official AutoML role names):** attribute = `capacity, region_id, latitude, longitude`; available_at_forecast = `temp, prcp, day_of_week, month, is_weekend`; unavailable_at_forecast = `avg_tripduration, pct_subscriber, ratio_gender`.
- **Weather:** single NYC GSOD station (LaGuardia `725030`/`14732`, verify in notebook), actuals as known-future proxy.
- **Holidays:** delegated to `holiday_region` config (AutoML `holiday_regions` / BQML `HOLIDAY_REGION`); not hand-engineered columns. (Exact region code, e.g. `US`, verified when wiring defaults.)
- **TimesFM:** pin `timesfm[torch]==2.0.1` which ships the **2.5** model classes; univariate v1, XReg opt-in later.

## Master Roadmap (Tier 1 — coarse)
- **Stage 1 — Data Architecture & Config Schema** — [in progress] — data layer, Pydantic/YAML config, GCP setup. (Detailed below.)
- **Stage 2 — Model Factory & Forecaster Abstractions** — [pending] — `src/geaptimes/models/`: `base.py` ABC, `factory.py`, `automl.py`, `timesfm.py` (CPU/GPU), `bqml.py`; local TimesFM inference demo.
- **Stage 3 — Experiment Tracking & DOE** — [pending] — DOE matrix engine in `utils/`; wire `aiplatform.Experiment`/`ExperimentRun`; per-run GCS path `gs://<bucket>/experiments/<name>/<timestamp>/`; log params + metrics.
- **Stage 4 — Managed Pipelines & Orchestration** — [pending] — KFP definitions in `src/geaptimes/pipelines/`; TimesFM cloud inference custom container (Model Garden→endpoint ref saved); side-by-side comparison DAG.
- **Stage 5 — Standardized Evaluation & Entry Point** — [pending] — uniform eval (MAE/RMSE/MAPE/Quantile Loss) over identical backtest windows; finalize `scripts/run_experiment.py` CLI.

## Stage 1 — Detailed Items (Tier 2)

### 1.1 Repo scaffold + standards + tracker
- `git init`; `uv init --package geaptimes` (creates `src/geaptimes/`); `uv python pin 3.11`.
- `pyproject.toml`: metadata, `requires-python>=3.11`, `[tool.ruff]` (line-length 100, target py311, `select=["ALL"]` with sensible ignores), `[tool.pytest.ini_options]` (`pythonpath=["src"]`), `[dependency-groups]` `dev=[ruff, ty]`, `test=[pytest, pytest-cov]`, `nb=[jupyter, matplotlib]`.
- Add runtime deps via `uv add`: `google-cloud-aiplatform==1.158.0`, `google-cloud-bigquery`, `db-dtypes`, `pandas`, `pydantic>=2`, `pyyaml`, `"timesfm[torch]==2.0.1"`. (`uv add`, never hand-edit deps.)
- `.pre-commit-config.yaml` from the skill template as-is (ruff + ty).
- Author docs: `CODE_STANDARDS.md` (uv/ruff/pytest+**ty**, no trailers, py3.11), `CLAUDE.md` (hyperlink → `CODE_STANDARDS.md` for all code/env changes), `README.md`, `.gitignore` (commit `uv.lock` — this is an application), `docs/notes/` dir.
- Update the `python-tooling` memory entry: type checker is `ty`, not mypy.
- **`PLANS.md`**: write this two-tier plan (Master Roadmap + Stage 1 items) as the living tracker.
- Commit; record hash against 1.1.

### 1.2 `config/schemas.py` — Pydantic v2 (the typed contract)
- Models: `ProjectConfig`, `WeatherConfig`, `SplitConfig`, `DateRange`, `CovariateRoles`, `DataConfig`, `ForecastSettings` (named to avoid clash with `timesfm.ForecastConfig`), `ExecutionConfig` (`Literal["local_cpu","local_gpu","cloud_pipeline"]`), and per-model params as a **discriminated union** on `params.type` (`TimesFMParams` / `BQMLParams` / `AutoMLParams`) with `Field(discriminator="type")`.
- Validators: `TimesFMParams.context_len` must be a multiple of 32; `horizon>0`; `top_n>0`.
- `ExperimentConfig.from_yaml(path)`: load YAML, `${ENV}` substitution, validate.
- `tests/test_schemas.py`: valid config parses; bad discriminator / non-multiple-of-32 `context_len` raise.

### 1.3 `config/base_config.yaml`
- The agreed DOE config: `project`, `data` (filters/splits/weather/covariate roles/holiday_region), `forecast` (`frequency:"D"`, `horizon:14`), `models` list (timesfm enabled univariate, bqml_arima_xreg enabled, automl disabled), `execution` (`target: local_cpu`). Must validate cleanly via 1.2.

### 1.4 `src/geaptimes/data/queries.py` + `utils/logger.py`
- Parameterized SQL template builders (functions returning SQL strings from a `DataConfig`): `build_source_query()` (top-N stations → daily agg + engineered covariates → per-station date spine + zero-fill → join GSOD weather + station metadata + calendar features) and `build_prepped_query()` (adds `splits`). Mirror statmike `_source`→`_prepped`.
- `utils/logger.py`: minimal structured logger used across the package.
- `tests/test_queries.py`: generated SQL contains expected table names / columns / `LIMIT {top_n}` (string-level, no BQ calls).

### 1.5 `data_notebooks/01_citibike_prep.ipynb`
- Loads config, builds `<dataset>.citibike_daily_source` then `_prepped` via `queries.py`.
- **Verifies in-notebook:** actual max date of the frozen Citibike table; LaGuardia GSOD station ids; `citibike_stations` name-join coverage (report unmatched). Visualize per-series ranges + split bands (statmike style).

### 1.6 `scripts/setup_gcp.py`
- Idempotent provisioning of the GCS bucket + BQ dataset from `ProjectConfig`; `--config` and `--dry-run` flags; reuse `utils/logger.py`. No-op if resources exist.

## Verification
- **No-cloud (CI-able):** `uv sync` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run ty check` → `uv run pytest` (schema + query-string tests pass) → `uv run python -c "from geaptimes... import ExperimentConfig; ExperimentConfig.from_yaml('config/base_config.yaml')"` loads/validates.
- **Cloud (needs auth + real project; manual):** `uv run python scripts/setup_gcp.py --config config/base_config.yaml --dry-run` then live; execute `01_citibike_prep.ipynb` to build tables and confirm row counts / split distribution.
- **Checkpoint:** STOP after Stage 1 — present SQL logic, the YAML+Pydantic schema, and table/split output for approval before Stage 2.

---

*Note: during execution the schema landed in the package as `src/geaptimes/schemas.py` (importable
as `geaptimes.schemas`) rather than `config/schemas.py`, and Stage 1 expanded to items 1.7–1.10
(notebook kernel standardization, data-layer hardening + live build, GCP resource labels + README
badges, and the `geaptimes.naming` table-naming convention). See `PLANS.md` for the as-built record.*
