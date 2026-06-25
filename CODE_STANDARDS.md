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

## Source layout

- `src/` layout: the importable package is `src/geaptimes/`. Import as `from geaptimes... import ...`.
- Config in `config/`, notebooks in `data_notebooks/`, CLI/ops scripts in `scripts/`,
  durable session notes in `docs/notes/`.

## Configuration

- Experiments are defined in **YAML** and parsed into **Pydantic v2** models
  (`src/geaptimes/schemas.py`) for type safety. Code consumes the validated `ExperimentConfig`,
  never raw YAML.
- The GCP project is supplied via the `GCP_PROJECT` env var (default `hybrid-vertex`); see
  `.env.example`. Do not hardcode project ids in committed config.

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
