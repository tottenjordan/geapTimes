# Post-Stage-5: eval-metric comparability audit + modern-python security hooks

## Context

Stage 5 is complete and merged to `main`. The user raised that **AutoML eval metrics haven't looked
comparable** in test runs, and asked to (a) investigate the eval-metric **components/steps**, and
(b) take on backlog items #2/#3/#4. After scoping, the user chose to **skip #2 (ParallelFor)** and
**#4 (AutoML tabular)** — both deferral rationales still hold — leaving **#3 (modern-python security
hooks)**. So this plan has two independent workstreams:

- **A. Eval-metric comparability audit + hardening** (the substantive one, motivated by the AutoML concern)
- **B. #3 security pre-commit hooks**

### Audit already performed (findings)

Traced the scoring path: every backend (BQML/AutoML in-process train→infer, TimesFM served) funnels
predictions into **one** `score_and_track_step` (`src/geaptimes/pipelines/steps.py:345`) →
`evaluate()` (`src/geaptimes/experiment/metrics.py:98`) → `compare_step`
(`steps.py:571`) → `rank_backends` (`src/geaptimes/experiment/comparison.py:24`). Confirmed live:
AutoML scored over `n_points=308`, **identical** to TimesFM/BQML — so **there is no alignment/scoring
bug**; the AutoML MAE gap (142 vs 78–85) is real (floor budget). But the audit surfaced genuine
**comparability-integrity gaps**:

1. **No cross-backend `n_points` parity check.** `evaluate()`/`_merge` (`metrics.py:134`) inner-joins
   and only raises on an *empty* overlap. A backend with *partial* overlap silently scores over fewer
   points, and `rank_backends` then compares metrics over **different denominators** — with nothing
   flagging it. `n_points` is computed per backend but never compared.
2. **`n_points` is invisible in the comparison.** `RANKING_METRICS` (`comparison.py:12`) =
   `[mae, rmse, smape, quantile_loss]`; the Markdown table never shows `n_points`, so a reviewer can't
   see whether backends were scored over the same points.
3. **Ranking is not NaN-safe.** `rank_backends` uses `metrics.get(m, inf)`; a present-but-`NaN` metric
   (the pre-5.2 shape) passes through as NaN and makes the `.sort()` order undefined.

## Workstream A — eval-metric comparability hardening

### A1. Surface + guard `n_points` — `src/geaptimes/experiment/comparison.py`
- Add an `n_points` column to the rendered ranking table (display only — **not** a sort key; ranking
  stays RMSE→MAE). Source it from each backend's metrics dict (already carried through
  `score_and_track` → `compare_backends`).
- Add a `warnings: list[str]` field to `Comparison`; in `rank_backends`, if the distinct non-zero
  `n_points` values across backends differ, append a warning ("backends scored over differing point
  counts: …") so partial-overlap comparability loss is loud, not silent. `render_ranking_markdown`
  prints any warnings under the table; `compare_step` logs them.

### A2. NaN-safe ranking — `comparison.py`
- Coerce non-finite metric values to `inf` in the ranking key (treat NaN/missing identically) so a
  degenerate metric sinks the model rather than corrupting the sort.

### A3. Partial-overlap visibility — `src/geaptimes/experiment/metrics.py`
- In `_merge`, when the inner-join overlap is smaller than the prediction rows for the covered series
  (i.e. predictions exist that found no actual, or vice-versa within the TEST window), `log` a warning
  with the counts. Keep the empty-overlap `ValueError` as-is. Pure/no behavior change to metrics.

### A4. Tests
- `tests/test_comparison.py` — `n_points` column rendered; parity warning fires on differing
  `n_points` and is absent when equal; NaN/inf metric sorts last; ranking order unchanged for the
  happy path.
- `tests/test_metrics.py` — partial-overlap warning emitted (via `caplog`) when some keys don't join;
  no warning on full overlap; empty overlap still raises.

### A5. Findings writeup — `docs/notes/automl-eval-metric-comparability.md` (new)
- Record the audit: shared-scorer path, the live `n_points=308` parity (no bug), the real AutoML
  quality gap at floor budget, the three integrity gaps + the hardening, and the still-open item that
  AutoML's full metric suite (sMAPE/quantile-loss) has **never been produced live** (all successful
  AutoML runs predate the 5.2 scorer; 5.5 ran `--disable-automl`).

## Workstream B — #3 modern-python security pre-commit hooks

Scoping finding: **no `.github/workflows/` and no shell scripts** exist, so `actionlint`/`zizmor`/
`shellcheck` have nothing to lint today — only `detect-secrets` is substantive now. `pre-commit` is
also not a declared dependency though `.pre-commit-config.yaml` assumes `uv run pre-commit`.

### B1. Declare tooling — `pyproject.toml`
- Add `pre-commit` and `detect-secrets` to the `dev` dependency group (both run via `uv`).

### B2. Hooks — `.pre-commit-config.yaml`
- Add **`detect-secrets`** with `args: ["--baseline", ".secrets.baseline"]` (active).
- Add **`shellcheck`**, **`actionlint`**, **`zizmor`** with a comment that they are **dormant until
  shell scripts / `.github/workflows/` exist** — wired so they fire the moment a target appears. Pin
  every new hook `rev`.

### B3. Baseline — `.secrets.baseline` (new)
- `uv run detect-secrets scan > .secrets.baseline`; review to confirm **no real secrets** (repo uses
  `${GCP_PROJECT}` substitution + resource *names*, not keys). Commit it.

### B4. Standards — `CODE_STANDARDS.md`
- Move the security-hooks line from "Deferred / backlog" into adopted tooling: `detect-secrets` active,
  the GH-Actions/shell linters configured-but-dormant.

## Housekeeping — `PLANS.md`
- Flip the stale Tier 1 row **Stage 5: `in progress` → `done`**.
- Add a short post-Stage-5 note: workstreams A + B done (hashes); #2 and #4 **declined**, deferrals
  reaffirmed with one-line rationale.

## Out of scope (decided this session)
- **#2 ParallelFor**, **#4 AutoML tabular / GCPC op** — skipped; current designs stay.
- **AutoML full-suite live confirmation run** — offered separately (needs go-ahead; ~2.5 h + AutoML
  budget). Would confirm AutoML's sMAPE/quantile-loss under the 5.2 scorer. Not required to land A/B.

## Verification
- Offline gate green: `uv run ruff check . && uv run ruff format --check . && uv run ty check &&
  uv run pytest` (218 existing + new tests).
- `uv run detect-secrets scan` yields a clean/audited baseline; `uv run pre-commit run --all-files`
  passes (dormant linters no-op). *If the environment restricts fetching hook repos, the config +
  baseline remain correct for CI; verify via `check-yaml` + the direct `detect-secrets` run.*

## Commit ritual
On `main` (default) → **branch first** (e.g. `post-stage5-eval-audit-and-tooling`). Separate commits:
(1) workstream A code + note, (2) workstream B tooling, then a **separate** `PLANS.md` commit recording
hashes + the Tier 1 flip. Push / PR only if the user asks.
