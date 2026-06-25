# geapTimes

A production-grade, **config-driven** time-series forecasting framework for **NYC Citibike**
demand, built on the **Gemini Enterprise Agent Platform (Vertex AI)**.

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

## Layout

```
config/          # base_config.yaml + Pydantic schemas (the typed contract)
data_notebooks/  # data exploration & prep
scripts/         # GCP setup, experiment CLI
src/geaptimes/   # package: data, models, pipelines, evaluation, utils
docs/notes/      # durable design notes
```
