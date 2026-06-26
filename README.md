<div align="center">

<h1 align="center">🚲 geapTimes ⌛</h1>

> A modular, production-grade factory framework for enterprise time-series forecasting on the **Gemini Enterprise Agent Platform** (GEAP, *fka Vertex AI*). The `ForecastFactory` creates model experiments for Google's `TimesFM`, `AutoML`, and `BigQuery ML` services.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/packaging-uv-DE5FE9?logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)
![ty](https://img.shields.io/badge/types-ty-261230?logo=astral&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-669DF6?logo=googlebigquery&logoColor=white)
![Vertex AI](https://img.shields.io/badge/Vertex%20AI-1.158-4285F4?logo=googlecloud&logoColor=white)
![TimesFM](https://img.shields.io/badge/TimesFM-2.5-FF6F00)

</div>

Forecasting logic is accessed through a **model factory** (AutoML, BQML `ARIMA_PLUS_XREG`, and
TimesFM 2.5). Experiments are defined in **YAML** and validated into **Pydantic** config. Runs
are tracked as platform Experiments with a Design-of-Experiments (DOE) matrix.

> Refactor of the notebook-based [`vertex-forecas-repo`](https://github.com/tottenjordan/vertex-forecas-repo)
> into a packaged framework. Built in staged milestones — see [PLANS.md](./PLANS.md).

## Data

- **Target:** daily trips per station — `COUNT(*)` of Citibike trips grouped by
  `start_station_name` and day (top-25 stations by volume).
- **Covariates:** NOAA GSOD weather (temp, precip), calendar features, and station metadata.
- Source: `bigquery-public-data` (Citibike + NOAA GSOD). See
  [docs/notes](./docs/notes/forecasting-covariate-and-model-constraints.md) for the covariate
  taxonomy and per-model constraints.

## Tech stack

- Python **3.11**, **`uv`** for packaging, **`ruff`** + **`ty`** + **`pytest`**.
- `google-cloud-aiplatform==1.158.0`, `timesfm[torch]==2.0.1` (loads the TimesFM **2.5** model).

## Quickstart

```bash
uv sync --all-groups          # create env + install deps
uv run ruff check .           # lint
uv run ty check               # type-check
uv run pytest                 # tests
```

See [CODE_STANDARDS.md](./CODE_STANDARDS.md) for the full development workflow.

## Notebooks (standardized kernel)

Notebooks run on a shared, reproducible Jupyter kernel backed by the project's `uv` venv:

```bash
# one-time per machine: register the kernel
uv run python -m ipykernel install --user --name geaptimes --display-name "geapTimes (uv 3.11)"
uv run jupyter lab            # notebooks auto-select the "geaptimes" kernel
```

The notebooks reference this kernel by name (`geaptimes`), so everyone runs the same env.

## GCP setup

```bash
cp .env.example .env          # defaults GCP_PROJECT=hybrid-vertex
set -a; source .env; set +a
gcloud auth application-default login
uv run python scripts/setup_gcp.py --config config/base_config.yaml --dry-run
```

> Want zero-setup sharing instead? The notebooks can be made Colab-friendly by adding a guarded
> first cell that `pip install`s the package and runs `google.colab.auth` — see CODE_STANDARDS.

## Repository structure

```
geapTimes/
├── config/
│   └── base_config.yaml              # the YAML experiment definition (validated into typed config)
├── src/geaptimes/                    # the installable package
│   ├── schemas.py                    # Pydantic v2 models — the typed config contract
│   ├── naming.py                     # deterministic dataset/table/model/bucket names
│   ├── constants.py                  # shared constants (resource labels, quantiles, …)
│   ├── gcp.py                        # GCP client/session helpers
│   ├── data/
│   │   └── queries.py                # BigQuery prep/train/infer SQL builders
│   ├── models/                       # the model factory + forecaster backends
│   │   ├── base.py                   # Forecaster ABC, ForecastResult, column constants
│   │   ├── factory.py                # ForecastFactory.from_config -> list[Forecaster]
│   │   ├── timesfm.py                # TimesFM 2.5 (in-process)
│   │   ├── bqml.py                   # BigQuery ML ARIMA_PLUS_XREG (server-side SQL)
│   │   └── automl.py                 # Vertex AutoML Forecasting (managed job; default-off)
│   ├── experiment/                   # the experiment-tracking + DOE harness
│   │   ├── doe.py                    # Design-of-Experiments matrix expansion
│   │   ├── metrics.py                # point metrics (MAE/RMSE) vs TEST actuals
│   │   ├── tracking.py               # Vertex AI Experiments + GCS artifact sink
│   │   └── runner.py                 # run_experiment: DOE × models -> tracked runs
│   └── utils/
│       └── logger.py                 # structured logging
├── data_notebooks/                   # demo notebooks (shared `geaptimes` kernel)
│   ├── 01_citibike_prep.ipynb        # data exploration & prep
│   ├── 02_timesfm_local.ipynb        # local TimesFM forecasting
│   └── 03_experiment_tracking.ipynb  # DOE + experiment-tracking demo
├── scripts/                          # GCP setup + standalone demos
│   ├── setup_gcp.py                  # provision datasets/buckets/labels
│   └── demo_timesfm.py
├── tests/                            # pytest suite (offline; cloud seams injected)
├── docs/
│   ├── plans/                        # immutable approved per-stage plans
│   └── notes/                        # durable cross-session design notes
├── PLANS.md                          # living roadmap + active-stage tracker
├── CODE_STANDARDS.md                 # authoritative tooling/layout/commit policy
└── pyproject.toml                    # uv-managed project + tool config
```

## Documentation & references

**Platform & models**

- **GEAP / Vertex AI AutoML Forecasting** —
  [overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/forecasting/overview)
- **BigQuery ML forecasting** —
  [overview](https://docs.cloud.google.com/bigquery/docs/forecasting-overview) ·
  [`ARIMA_PLUS_XREG` model syntax](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-time-series) ·
  [multivariate forecasting tutorial](https://docs.cloud.google.com/bigquery/docs/arima-plus-xreg-single-time-series-forecasting-tutorial)
- **TimesFM 2.5** —
  [HuggingFace model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) ·
  [GitHub repo](https://github.com/google-research/timesfm)
- **Vertex AI Experiments** (run tracking / params / metrics) —
  [docs](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments)
- **Public datasets** —
  [Citibike trips](https://console.cloud.google.com/marketplace/details/city-of-new-york/nyc-citi-bike) ·
  [NOAA GSOD weather](https://console.cloud.google.com/marketplace/details/noaa-public/gsod)

**Tooling**

- [`uv`](https://docs.astral.sh/uv/) packaging ·
  [`ruff`](https://docs.astral.sh/ruff/) lint+format ·
  [`ty`](https://github.com/astral-sh/ty) type checker ·
  [Pydantic v2](https://docs.pydantic.dev/latest/) ·
  [pytest](https://docs.pytest.org/)
