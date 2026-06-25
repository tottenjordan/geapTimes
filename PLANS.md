# geapTimes — Plans & Progress

Living tracker. **Tier 1** is a coarse master roadmap of all stages; **Tier 2** decomposes the
*active* stage into numbered items with status and commit hash. Later stages are expanded
just-in-time when reached. Item IDs are stable and map to commits.

Status legend: `pending` · `in progress` · `done` · `blocked`

---

## Tier 1 — Master Roadmap

| Stage | Title | Status |
|-------|-------|--------|
| 1 | Data Architecture & Config Schema | items done — STOP (awaiting approval) |
| 2 | Model Factory & Forecaster Abstractions | pending |
| 3 | Experiment Tracking & DOE Framework | pending |
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

## Tier 2 — Stage 1 (Data Architecture & Config Schema)

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
| 1.9 | GCP resource labels (`solution=geaptimes`) + README badges | in progress | — |

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

### Stage 1 verification
- **No-cloud:** `uv sync` → `uv run ruff check .` → `uv run ruff format --check .` →
  `uv run ty check` → `uv run pytest` → load `base_config.yaml` via `ExperimentConfig.from_yaml`.
- **Cloud (manual, needs auth):** `setup_gcp.py --dry-run` then live; run the prep notebook.
- **STOP** checkpoint: present SQL logic, YAML+Pydantic schema, and table/split output.
