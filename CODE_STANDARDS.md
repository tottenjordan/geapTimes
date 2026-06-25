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

- Experiments are defined in **YAML** and parsed into **Pydantic v2** models (`config/schemas.py`)
  for type safety. Code consumes the validated `ExperimentConfig`, never raw YAML.

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
