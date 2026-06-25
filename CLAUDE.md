# CLAUDE.md

Guidance for working in the **geapTimes** repository.

## Code standards — read first

**Always refer to [CODE_STANDARDS.md](./CODE_STANDARDS.md) when writing code or making
environment changes.** It is authoritative for Python version, tooling (`uv`, `ruff`, `ty`,
`pytest`), source layout, configuration, and git/commit policy.

Key non-negotiables (see the standards doc for the full list):

- Use **`uv`** for all package management and command execution (`uv add`, `uv run …`); never
  bare `pip`/`python`.
- **`ruff`** for lint+format, **`ty`** for type checking, **`pytest`** for tests.
- **Never** add `Co-Authored-By` or "Generated with …" trailers to commits or PRs.

## Project

geapTimes is a config-driven time-series forecasting framework (NYC Citibike demand) on the
Gemini Enterprise Agent Platform (Vertex AI). Forecasting logic is accessed via a model
factory; experiments are defined in YAML and validated into typed config.

## Planning & progress

- **[PLANS.md](./PLANS.md)** is the living tracker: a coarse master roadmap plus the active
  stage decomposed into numbered items with status and commit hashes.
- Work proceeds in staged milestones, each ending in a STOP checkpoint for approval.

## Notes

- Durable, cross-session insights that aren't recoverable from the repo go in `docs/notes/`,
  organized by topic.
