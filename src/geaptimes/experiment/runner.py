"""The experiment run loop: expand the DOE grid, run each model, and track every run.

``run_experiment`` is the Stage 3 entry point. For each :class:`~geaptimes.experiment.doe.DOEPoint`
(a re-validated config variant) it builds the enabled forecasters, opens a tracked run, executes
``fit()`` → ``predict()``, scores the point forecast against the held-out TEST actuals, and logs
params, metrics, and artifacts. This is the first place the Stage 2 forecasters execute for real.

All cloud-touching collaborators — the tracker, the forecaster factory, the actuals loader, and the
clock — are injectable, so the orchestration unit-tests offline with fakes.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from geaptimes.experiment.doe import expand, point_slug
from geaptimes.experiment.metrics import point_metrics
from geaptimes.experiment.tracking import ExperimentTracker
from geaptimes.models.factory import ForecastFactory
from geaptimes.naming import table_names
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from geaptimes.models.base import Forecaster
    from geaptimes.schemas import ExperimentConfig

logger = get_logger(__name__)

ForecasterBuilder = Callable[["ExperimentConfig"], "list[Forecaster]"]
ActualsLoader = Callable[["ExperimentConfig"], "pd.DataFrame"]


@dataclass
class RunRecord:
    """Outcome of one tracked run (one DOE point x one model)."""

    run_name: str
    model: str
    overrides: dict[str, Any]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_uri: str = ""


def run_experiment(
    cfg: "ExperimentConfig",
    *,
    tracker: ExperimentTracker | None = None,
    forecasters: ForecasterBuilder | None = None,
    actuals_loader: ActualsLoader | None = None,
    now: Callable[[], datetime] | None = None,
) -> list[RunRecord]:
    """Run every (DOE variant x enabled model), tracking params/metrics/artifacts for each."""
    tracker = tracker or ExperimentTracker(cfg)
    build = forecasters or ForecastFactory.from_config
    load_actuals = actuals_loader or _default_actuals_loader
    clock = now or _utc_now

    records: list[RunRecord] = []
    for point in expand(cfg):
        slug = point_slug(point.overrides)
        actuals = load_actuals(point.config)
        for forecaster in build(point.config):
            run_name = _safe_run_name(
                f"{forecaster.name}-{slug}-{clock().strftime('%Y%m%d-%H%M%S')}"
            )
            logger.info("running %s", run_name)
            with tracker.run(run_name):
                forecaster.fit()
                result = forecaster.predict()
                data = point.config.data
                metrics = point_metrics(
                    result.predictions,
                    actuals,
                    series_col=data.series_column,
                    time_col=data.time_column,
                    target_col=data.target_column,
                )
                tracker.log_params(_run_params(point.config, forecaster, point.overrides))
                tracker.log_metrics(metrics)
                artifact_uri = tracker.log_artifacts(
                    run_name, config=point.config, predictions=result.predictions
                )
            records.append(
                RunRecord(
                    run_name=run_name,
                    model=forecaster.name,
                    overrides=dict(point.overrides),
                    metrics=metrics,
                    metadata=dict(result.metadata),
                    artifact_uri=artifact_uri,
                )
            )
    return records


def _run_params(
    cfg: "ExperimentConfig", forecaster: "Forecaster", overrides: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the params logged for a run: model, backend, forecast settings, DOE overrides."""
    params: dict[str, Any] = {
        "model": forecaster.name,
        "backend": forecaster.model_cfg.params.type,
        "horizon": cfg.forecast.horizon,
        "target": cfg.data.target_column,
        "top_n": cfg.data.station_filter.top_n,
    }
    params.update({f"doe.{path}": value for path, value in overrides.items()})
    return params


def _default_actuals_loader(cfg: "ExperimentConfig") -> "pd.DataFrame":
    """Read the held-out TEST rows (series/time/target) from the prepped table in BigQuery."""
    from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

    data = cfg.data
    table = table_names(cfg)["prepped"]
    prepped = f"{cfg.project.id}.{cfg.project.bq_dataset}.{table}"
    cols = f"{data.series_column}, {data.time_column}, {data.target_column}"
    # SQL built from trusted config (column/table names), not user input.
    sql = f"SELECT {cols} FROM `{prepped}` WHERE splits = 'TEST'"  # noqa: S608
    client = bigquery.Client(project=cfg.project.id)
    return client.query(sql).to_dataframe()


def _safe_run_name(name: str) -> str:
    """Vertex run ids allow only lowercase letters, digits, and hyphens."""
    return "".join(ch if ch.isalnum() else "-" for ch in name.lower())


def _utc_now() -> datetime:
    return datetime.now(UTC)
