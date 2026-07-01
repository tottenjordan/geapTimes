"""Cross-backend ranking and comparison-report rendering.

The single place backends are ranked into a winner and rendered as a Markdown table, shared by the
pipeline's ``compare-backends`` step (via :func:`~geaptimes.pipelines.steps.compare_step`) and the
``run_experiment`` CLI. Ranking is by RMSE (tie-break MAE) — the lowest-error model wins — while the
table surfaces the full standardized metric set so the richer metrics are legible at a glance.

The table also surfaces ``n_points`` (the number of backtest points each backend was scored over) so
a reviewer can see at a glance whether the ranked metrics share a denominator. When they *don't*
match, :func:`rank_backends` records a comparability warning rather than ranking silently over
differing point counts — an inner-join scorer can otherwise compare metrics computed over different
supports without anything flagging it.
"""

import math
from dataclasses import dataclass, field
from typing import Any

# Metrics shown in the ranking table, in column order. Ranking still keys on RMSE then MAE.
RANKING_METRICS: list[str] = ["mae", "rmse", "smape", "quantile_loss"]

# Reporting metrics shown after the ranking metrics: the statmike-parity SQL set (classic `mape`,
# `mse`) plus the demand-normalized `pmae`/`prmse`. Display-only, 4-decimal formatted, never sort
# keys — they let results line up with the statmike reference notebook and read quality relative to
# demand. (`mape` uses SAFE_DIVIDE semantics and may cover fewer points than `n_points`.)
_DISPLAY_REPORT_METRICS: list[str] = ["mape", "mse", "pmae", "prmse"]

# Integer-formatted count column; display-only.
_DISPLAY_COUNT_METRICS: list[str] = ["n_points"]

# All display-only columns appended after the ranking metrics, in render order.
_DISPLAY_METRICS: list[str] = [*_DISPLAY_REPORT_METRICS, *_DISPLAY_COUNT_METRICS]


@dataclass
class Comparison:
    """The side-by-side comparison outcome: the ranked rows, the winner, and any warnings.

    ``warnings`` holds comparability advisories (e.g. backends scored over differing point counts);
    it is empty on the clean path. Consumers surface it — :func:`render_ranking_markdown` prints it
    under the table and :func:`~geaptimes.pipelines.steps.compare_step` logs it.
    """

    ranking: list[dict[str, Any]] = field(default_factory=list)
    winner: str = ""
    warnings: list[str] = field(default_factory=list)


def _rank_value(value: float) -> float:
    """Coerce a metric to a sort key: non-finite (NaN/inf/missing) sinks to ``+inf``."""
    return value if math.isfinite(value) else float("inf")


def rank_backends(results: "list[tuple[str, dict[str, float]]]") -> Comparison:
    """Rank ``(model, metrics)`` pairs by RMSE (tie-break MAE); the lowest-error model wins.

    Each ranking row carries the full :data:`RANKING_METRICS` set plus display-only ``n_points``
    (missing metrics default to ``inf``), so the renderer and downstream consumers see one uniform
    shape regardless of which backend produced them. The sort is NaN-safe: a present-but-non-finite
    metric is treated as ``+inf`` so it sinks the model rather than corrupting the order. If
    backends were scored over differing (non-zero) ``n_points`` a comparability warning is
    recorded, since the ranked metrics would then span different denominators.
    """
    ranking: list[dict[str, Any]] = [
        {
            "model": name,
            **{m: float(metrics.get(m, float("inf"))) for m in RANKING_METRICS},
            **{m: float(metrics.get(m, float("nan"))) for m in _DISPLAY_METRICS},
        }
        for name, metrics in results
    ]
    ranking.sort(key=lambda r: (_rank_value(r["rmse"]), _rank_value(r["mae"])))
    winner = str(ranking[0]["model"]) if ranking else ""

    warnings = _parity_warnings(ranking)
    return Comparison(ranking=ranking, winner=winner, warnings=warnings)


def _parity_warnings(ranking: "list[dict[str, Any]]") -> list[str]:
    """Return a comparability warning if backends were scored over differing point counts."""
    counts = {
        int(row["n_points"])
        for row in ranking
        if math.isfinite(row["n_points"]) and row["n_points"] > 0
    }
    if len(counts) <= 1:
        return []
    detail = ", ".join(
        f"{row['model']}={int(row['n_points'])}"
        for row in ranking
        if math.isfinite(row["n_points"]) and row["n_points"] > 0
    )
    return [
        f"backends scored over differing point counts ({detail}); "
        f"ranked metrics span different denominators and are not directly comparable"
    ]


def render_ranking_markdown(comparison: Comparison) -> str:
    """Render a :class:`Comparison` as a Markdown table, winner flagged, lowest RMSE first.

    Any comparability :attr:`Comparison.warnings` are printed as a bulleted note under the table.
    """
    header = ["rank", "model", *RANKING_METRICS, *_DISPLAY_METRICS]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for idx, row in enumerate(comparison.ranking, start=1):
        flag = " (winner)" if row["model"] == comparison.winner else ""
        metric_cells = [_format_metric(row[m]) for m in RANKING_METRICS]
        report_cells = [_format_metric(row[m]) for m in _DISPLAY_REPORT_METRICS]
        count_cells = [_format_count(row[m]) for m in _DISPLAY_COUNT_METRICS]
        cells = [str(idx), f"{row['model']}{flag}", *metric_cells, *report_cells, *count_cells]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"

    if comparison.warnings:
        notes = "\n".join(f"- ⚠️ {w}" for w in comparison.warnings)
        table += "\n" + notes + "\n"
    return table


def _format_metric(value: float) -> str:
    """Format an error-metric cell to 4 decimals (``nan`` for non-finite values)."""
    return f"{value:.4f}" if math.isfinite(value) else "nan"


def _format_count(value: float) -> str:
    """Format a count cell (``n_points``) as an integer (``nan`` for non-finite values)."""
    return str(int(value)) if math.isfinite(value) else "nan"
