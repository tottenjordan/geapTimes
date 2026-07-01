"""Cross-backend ranking and comparison-report rendering.

The single place backends are ranked into a winner and rendered as a Markdown table, shared by the
pipeline's ``compare-backends`` step (via :func:`~geaptimes.pipelines.steps.compare_step`) and the
``run_experiment`` CLI. Ranking is by RMSE (tie-break MAE) — the lowest-error model wins — while the
table surfaces the full standardized metric set so the richer metrics are legible at a glance.
"""

from dataclasses import dataclass, field
from typing import Any

# Metrics shown in the ranking table, in column order. Ranking still keys on RMSE then MAE.
RANKING_METRICS: list[str] = ["mae", "rmse", "smape", "quantile_loss"]


@dataclass
class Comparison:
    """The side-by-side comparison outcome: the ranked rows and the winning model."""

    ranking: list[dict[str, Any]] = field(default_factory=list)
    winner: str = ""


def rank_backends(results: "list[tuple[str, dict[str, float]]]") -> Comparison:
    """Rank ``(model, metrics)`` pairs by RMSE (tie-break MAE); the lowest-error model wins.

    Each ranking row carries the full :data:`RANKING_METRICS` set (missing metrics default to
    ``inf``), so the renderer and downstream consumers see one uniform shape regardless of which
    backend produced them.
    """
    ranking: list[dict[str, Any]] = [
        {"model": name, **{m: float(metrics.get(m, float("inf"))) for m in RANKING_METRICS}}
        for name, metrics in results
    ]
    ranking.sort(key=lambda r: (r["rmse"], r["mae"]))
    winner = str(ranking[0]["model"]) if ranking else ""
    return Comparison(ranking=ranking, winner=winner)


def render_ranking_markdown(comparison: Comparison) -> str:
    """Render a :class:`Comparison` as a Markdown table, winner flagged, lowest RMSE first."""
    header = ["rank", "model", *RANKING_METRICS]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for idx, row in enumerate(comparison.ranking, start=1):
        flag = " (winner)" if row["model"] == comparison.winner else ""
        cells = [str(idx), f"{row['model']}{flag}", *(f"{row[m]:.4f}" for m in RANKING_METRICS)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
