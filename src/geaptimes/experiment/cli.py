"""CLI for the in-process experiment runner.

Thin orchestration over :func:`~geaptimes.experiment.runner.run_experiment` (the Stage 3 in-process
path): load + validate the YAML config, optionally flip the AutoML backend on/off for this run, then
either list the planned runs (``--dry-run``, no cloud) or execute them and print a per-run metrics
table plus a ranked comparison report (the same :mod:`~geaptimes.experiment.comparison` renderer the
pipeline uses). The runner is injected so this orchestration unit-tests offline.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from geaptimes.experiment.comparison import RANKING_METRICS, rank_backends, render_ranking_markdown
from geaptimes.experiment.doe import expand, point_slug
from geaptimes.experiment.overrides import with_automl_disabled, with_automl_enabled
from geaptimes.experiment.runner import RunRecord, run_experiment
from geaptimes.schemas import ExperimentConfig
from geaptimes.utils.logger import get_logger

logger = get_logger(__name__)

Runner = Callable[[ExperimentConfig], list[RunRecord]]


def planned_runs(cfg: ExperimentConfig) -> list[tuple[str, str]]:
    """Enumerate the ``(doe_slug, model)`` runs that would execute (no cloud calls)."""
    return [
        (point_slug(point.overrides), model.name)
        for point in expand(cfg)
        for model in point.config.models
        if model.enabled
    ]


def format_records_table(records: list[RunRecord]) -> str:
    """Render the per-run metrics as an aligned text table (run, model, then the metric suite)."""
    headers = ["run", "model", *RANKING_METRICS]
    rows = [
        [
            rec.run_name,
            rec.model,
            *(f"{rec.metrics.get(m, float('nan')):.4f}" for m in RANKING_METRICS),
        ]
        for rec in records
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def build_comparison_report(records: list[RunRecord]) -> str:
    """Rank the runs (RMSE then MAE) and render the shared Markdown comparison table."""
    return render_ranking_markdown(rank_backends([(rec.model, rec.metrics) for rec in records]))


def main(argv: "list[str] | None" = None, *, runner: Runner = run_experiment) -> int:
    """CLI: run every (DOE variant x enabled model) for a config and report, or ``--dry-run``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    automl_group = parser.add_mutually_exclusive_group()
    automl_group.add_argument(
        "--enable-automl",
        action="store_true",
        help="Enable the AutoML backend for this run (overrides the config).",
    )
    automl_group.add_argument(
        "--disable-automl",
        action="store_true",
        help="Disable the AutoML backend for this run (overrides the config) to skip the long, "
        "billable AutoML training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the planned (DOE variant x model) runs only; do not execute (no cloud calls).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the Markdown comparison report to this path (optional).",
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.enable_automl:
        cfg = with_automl_enabled(cfg)
    elif args.disable_automl:
        cfg = with_automl_disabled(cfg)

    if args.dry_run:
        runs = planned_runs(cfg)
        exp = cfg.execution.experiment_name
        logger.info("planned %d run(s) for experiment %s:", len(runs), exp)
        for slug, model in runs:
            logger.info("  %s x %s", slug, model)
        return 0

    records = runner(cfg)
    report = build_comparison_report(records)
    print(format_records_table(records))  # noqa: T201 - CLI results to stdout
    print()  # noqa: T201
    print(report)  # noqa: T201
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        logger.info("wrote comparison report to %s", args.output)
    return 0
