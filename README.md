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

## Layout

```
config/          # base_config.yaml + Pydantic schemas (the typed contract)
data_notebooks/  # data exploration & prep
scripts/         # GCP setup, experiment CLI
src/geaptimes/   # package: data, models, pipelines, evaluation, utils
docs/notes/      # durable design notes
```
