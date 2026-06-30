"""Pipeline step functions — the real orchestration logic, independent of KFP.

Each function does one unit of the comparison pipeline (build tables, run a backend, register /
deploy / predict / tear down the TimesFM serving model, compare results). The KFP components in
``components.py`` are thin shells that call these. Every cloud collaborator (the ``aiplatform``
module, BigQuery readers, GCS IO) is an injected seam with a lazy live default, so the orchestration
unit-tests offline with fakes — the same discipline as the Stage 2/3 forecasters and tracker.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from geaptimes.data.queries import (
    build_infer_query,
    build_prepped_query,
    build_source_query,
    build_train_query,
)
from geaptimes.experiment.comparison import Comparison, rank_backends
from geaptimes.experiment.metrics import evaluate
from geaptimes.experiment.runner import (
    RunRecord,
    _default_actuals_loader,
    _safe_run_name,
    _utc_now,
)
from geaptimes.models.base import (
    DATE_COLUMN,
    FORECAST_COLUMN,
    HORIZON_STEP_COLUMN,
    MODEL_COLUMN,
    PREDICTION_COLUMNS,
    QUANTILES,
    SERIES_COLUMN,
    quantile_column,
)
from geaptimes.models.factory import ForecastFactory
from geaptimes.naming import config_fingerprint_hash, table_names
from geaptimes.pipelines.config import (
    pipeline_root,
    resource_labels,
    timesfm_context_len,
    timesfm_endpoint_display_name,
    timesfm_model_display_name,
)
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from datetime import datetime

    import pandas as pd

    from geaptimes.experiment.tracking import ExperimentTracker
    from geaptimes.models.base import Forecaster
    from geaptimes.schemas import ExperimentConfig

logger = get_logger(__name__)


@dataclass
class BackendResult:
    """A backend run's tracked record plus its prediction frame (for artifacts + comparison)."""

    record: RunRecord
    predictions: "pd.DataFrame"


@dataclass(frozen=True)
class TableRef:
    """A BigQuery table's fully-qualified name + row count, for lineage Dataset artifacts."""

    name: str
    rows: int


# -- helpers --------------------------------------------------------------------------------------
def _qualified(cfg: "ExperimentConfig", role: str) -> str:
    """Fully-qualified ``project.dataset.table`` for a naming role."""
    return f"{cfg.project.id}.{cfg.project.bq_dataset}.{table_names(cfg)[role]}"


def _build_forecaster(cfg: "ExperimentConfig", model_name: str) -> "Forecaster":
    """Construct the named (enabled or not) model's forecaster via the factory."""
    for model_cfg in cfg.models:
        if model_cfg.name == model_name:
            return ForecastFactory.create(model_cfg, cfg)
    msg = f"model {model_name!r} not found in config"
    raise ValueError(msg)


def _lazy_aiplatform() -> Any:  # noqa: ANN401 - external SDK module
    from google.cloud import aiplatform  # noqa: PLC0415 - lazy cloud import

    return aiplatform


def _aip_init(aip: Any, cfg: "ExperimentConfig") -> None:  # noqa: ANN401 - external SDK module
    aip.init(
        project=cfg.project.id,
        location=cfg.project.region,
        staging_bucket=f"gs://{cfg.project.gcs_bucket}",
    )


def series_contexts(
    frame: "pd.DataFrame", cfg: "ExperimentConfig"
) -> "tuple[list[Any], list[list[float]], list[Any]]":
    """Per-series serving inputs: ``(series_ids, contexts, last_dates)`` from non-TEST history."""
    import pandas as pd  # noqa: PLC0415 - lazy heavy import

    data = cfg.data
    context_len = timesfm_context_len(cfg)
    train = frame[frame["splits"] != "TEST"]
    series_ids: list[Any] = []
    contexts: list[list[float]] = []
    last_dates: list[Any] = []
    for series_id, group in train.groupby(data.series_column, sort=True):
        ordered = group.sort_values(data.time_column)
        history = ordered[data.target_column].to_numpy(dtype=float)[-context_len:]
        series_ids.append(series_id)
        contexts.append([float(x) for x in history])
        last_dates.append(pd.Timestamp(ordered[data.time_column].iloc[-1]))
    return series_ids, contexts, last_dates


def responses_to_frame(
    responses: "list[dict[str, Any]]",
    last_date_by_series: "dict[Any, Any]",
    *,
    model_name: str,
) -> "pd.DataFrame":
    """Map served forecast responses onto the standardized long prediction frame.

    Each response is ``{"series_id", "horizon", "forecast": [...], "q10": [...], ...}``. Dates are
    keyed by series (not position), so this is correct even if batch prediction reorders rows.
    """
    import pandas as pd  # noqa: PLC0415 - lazy heavy import

    rows: list[dict[str, Any]] = []
    for resp in responses:
        series_id = resp["series_id"]
        last_date = last_date_by_series[series_id]
        horizon = int(resp["horizon"])
        for step in range(horizon):
            row: dict[str, Any] = {
                SERIES_COLUMN: series_id,
                DATE_COLUMN: (last_date + pd.Timedelta(days=step + 1)).date(),
                HORIZON_STEP_COLUMN: step + 1,
                FORECAST_COLUMN: float(resp[FORECAST_COLUMN][step]),
                MODEL_COLUMN: model_name,
            }
            for level in QUANTILES:
                row[quantile_column(level)] = float(resp[quantile_column(level)][step])
            rows.append(row)
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _default_frame_loader(cfg: "ExperimentConfig") -> "pd.DataFrame":  # pragma: no cover - live
    """Read series/time/target/splits from the prepped table (lazy BQ client)."""
    from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

    data = cfg.data
    prepped = _qualified(cfg, "prepped")
    cols = f"{data.series_column}, {data.time_column}, {data.target_column}, splits"
    sql = f"SELECT {cols} FROM `{prepped}`"  # noqa: S608 - trusted config identifiers
    return bigquery.Client(project=cfg.project.id).query(sql).to_dataframe()


# -- steps ----------------------------------------------------------------------------------------
def ensure_source_step(
    cfg: "ExperimentConfig",
    *,
    table_inspector: "Callable[[str], str | None] | None" = None,
    query_runner: "Callable[[str], Any] | None" = None,
    row_counter: "Callable[[str], int] | None" = None,
    force: bool = False,
) -> TableRef:
    """Build the daily ``source`` table if absent or stale; return its :class:`TableRef`.

    Self-bootstrapping front step: the pipeline no longer assumes an out-of-band Stage 1 notebook
    ran first. The guard is *existence + fingerprint* — ``table_inspector`` returns the fingerprint
    stored in the table's description (``None`` when the table does not exist), and a rebuild runs
    only when it is missing or differs from the current :func:`config_fingerprint_hash`. Since the
    frozen public window makes ``source`` effectively static, the common case is a cheap skip (a
    naive ``CREATE OR REPLACE`` would re-bill the multi-GB scan every run).
    """
    inspect = table_inspector or (lambda name: _table_fingerprint(cfg, name))
    run = query_runner or (lambda sql: _run_bq(cfg, sql))
    count = row_counter or (lambda name: _count_rows(cfg, name))
    source = _qualified(cfg, "source")
    fingerprint = config_fingerprint_hash(cfg)
    if not force and inspect(source) == fingerprint:
        logger.info("source table %s current (fingerprint %s) - skip rebuild", source, fingerprint)
        return TableRef(source, count(source))
    logger.info("building source table %s", source)
    run(build_source_query(cfg.data, source, description=fingerprint))
    return TableRef(source, count(source))


def ensure_prepped_step(
    cfg: "ExperimentConfig",
    *,
    table_inspector: "Callable[[str], str | None] | None" = None,
    query_runner: "Callable[[str], Any] | None" = None,
    row_counter: "Callable[[str], int] | None" = None,
    force: bool = False,
) -> TableRef:
    """Build the ``prepped`` table (adds the ``splits`` column) from ``source`` if absent or stale.

    Same existence + fingerprint guard as :func:`ensure_source_step`. ``prepped`` is split apart
    from ``source`` because it changes with covariate/split config while ``source`` changes ~never,
    so the two cache independently and the lineage chain (source -> prepped -> {train, infer,
    timesfm}) stays clean.
    """
    inspect = table_inspector or (lambda name: _table_fingerprint(cfg, name))
    run = query_runner or (lambda sql: _run_bq(cfg, sql))
    count = row_counter or (lambda name: _count_rows(cfg, name))
    source = _qualified(cfg, "source")
    prepped = _qualified(cfg, "prepped")
    fingerprint = config_fingerprint_hash(cfg)
    if not force and inspect(prepped) == fingerprint:
        logger.info("prepped table %s current (fingerprint %s) - skip", prepped, fingerprint)
        return TableRef(prepped, count(prepped))
    logger.info("building prepped table %s from %s", prepped, source)
    run(build_prepped_query(cfg.data, source, prepped, description=fingerprint))
    return TableRef(prepped, count(prepped))


def _table_fingerprint(  # pragma: no cover - live
    cfg: "ExperimentConfig", table: str
) -> str | None:
    """Read a table's stored fingerprint (its description), or ``None`` if it does not exist."""
    from google.api_core.exceptions import NotFound  # noqa: PLC0415 - lazy cloud import
    from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

    client = bigquery.Client(project=cfg.project.id)
    try:
        return client.get_table(table).description
    except NotFound:
        return None


def build_tables_step(
    cfg: "ExperimentConfig",
    *,
    query_runner: "Callable[[str], Any] | None" = None,
    row_counter: "Callable[[str], int] | None" = None,
) -> dict[str, TableRef]:
    """Build the train + infer tables from the prepped table; return resolved table refs.

    Each :class:`TableRef` carries the fully-qualified name + row count so the ``build_tables``
    component can emit a lineage ``Dataset`` artifact per table (``bq://`` uri + fingerprint +
    rows). ``prepped`` is produced (and its artifact emitted) upstream by
    :func:`ensure_prepped_step` and flows in as the lineage edge; train/infer are cheap CTAS views
    rebuilt every run.
    """
    run = query_runner or (lambda sql: _run_bq(cfg, sql))
    count = row_counter or (lambda name: _count_rows(cfg, name))
    prepped = _qualified(cfg, "prepped")
    train = _qualified(cfg, "train")
    infer = _qualified(cfg, "infer")
    logger.info("building train table %s", train)
    run(build_train_query(cfg.data, prepped, train))
    logger.info("building infer table %s", infer)
    run(build_infer_query(cfg.data, prepped, infer))
    return {
        "train": TableRef(train, count(train)),
        "infer": TableRef(infer, count(infer)),
    }


def _run_bq(cfg: "ExperimentConfig", sql: str) -> Any:  # noqa: ANN401  # pragma: no cover - live
    """Execute DDL/SQL on BigQuery via a lazily-built client (default for build_tables_step)."""
    from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

    return bigquery.Client(project=cfg.project.id).query(sql).result()


def _count_rows(cfg: "ExperimentConfig", table: str) -> int:  # pragma: no cover - live
    """Row count for a fully-qualified table (default ``row_counter`` for build_tables_step)."""
    from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

    sql = f"SELECT COUNT(*) AS n FROM `{table}`"  # noqa: S608 - trusted config identifiers
    return int(next(iter(bigquery.Client(project=cfg.project.id).query(sql).result())).n)


def train_backend_step(
    cfg: "ExperimentConfig",
    model_name: str,
    *,
    forecaster: "Forecaster | None" = None,
) -> str:
    """Train one backend and return its :attr:`~geaptimes.models.base.Forecaster.model_reference`.

    The reference (a BQML model id, a Vertex ``Model`` resource name, ``""`` for zero-shot) is the
    only thing the split needs to pass to :func:`infer_backend_step`: it lets the (separate) infer
    pod re-use the trained model, so an inference-side failure never forces a re-train. No tracking
    happens here — params/metrics/artifacts are logged once, in :func:`score_and_track_step`.
    """
    fc = forecaster or _build_forecaster(cfg, model_name)
    logger.info("training backend %s", fc.name)
    fc.fit()
    return fc.model_reference


def infer_backend_step(
    cfg: "ExperimentConfig",
    model_name: str,
    model_reference: str,
    *,
    forecaster: "Forecaster | None" = None,
) -> "pd.DataFrame":
    """Infer with a backend trained in a prior step, attached by ``model_reference``.

    Builds the forecaster, attaches the already-trained model (a no-op for backends whose
    ``predict`` is self-contained, e.g. BQML referencing the model by config-derived id), predicts,
    and returns the standardized long frame. Scoring + tracking are deferred to
    :func:`score_and_track_step`, the single tracking point for every backend.
    """
    fc = forecaster or _build_forecaster(cfg, model_name)
    fc.attach_model(model_reference)
    logger.info("inferring backend %s from %s", fc.name, model_reference or "<config>")
    return fc.predict().predictions


def _model_run_params(cfg: "ExperimentConfig", model_name: str) -> dict[str, Any]:
    """Params logged for any backend run, derived from the named model's config (no forecaster)."""
    backend = next(
        (m.params.type for m in cfg.models if m.name == model_name),
        model_name,
    )
    return {
        "model": model_name,
        "backend": backend,
        "horizon": cfg.forecast.horizon,
        "target": cfg.data.target_column,
        "top_n": cfg.data.station_filter.top_n,
    }


def score_and_track_step(  # noqa: PLR0913 - cfg + name + frame + injected seams (offline-testable)
    cfg: "ExperimentConfig",
    model_name: str,
    predictions: "pd.DataFrame",
    *,
    actuals_loader: "Callable[[ExperimentConfig], pd.DataFrame] | None" = None,
    tracker: "ExperimentTracker | None" = None,
    now: "Callable[[], datetime] | None" = None,
) -> BackendResult:
    """Score a backend's predictions against the TEST actuals and track the run.

    The single, shared tail of every backend chain: the in-process backends (train → infer) and the
    TimesFM serving chain (register → deploy → predict) all feed their predictions here, so each
    produces the same metrics + ``ExperimentRun`` + artifacts and the comparison consumes them
    identically. Params are derived from the model config (:func:`_model_run_params`) rather than a
    live forecaster, since this pod only has the predictions, not the trained model.
    """
    clock = now or _utc_now
    load_actuals = actuals_loader or _default_actuals_loader
    data = cfg.data
    run_name = _safe_run_name(f"{model_name}-{clock().strftime('%Y%m%d-%H%M%S')}")
    actuals = load_actuals(cfg)
    evaluation = evaluate(
        predictions,
        actuals,
        series_col=data.series_column,
        time_col=data.time_column,
        target_col=data.target_column,
    )
    metrics = evaluation.summary()
    logger.info("scoring backend %s as %s", model_name, run_name)
    artifact_uri = ""
    if tracker is not None:
        with tracker.run(run_name):
            tracker.log_params(_model_run_params(cfg, model_name))
            tracker.log_metrics(metrics)
            artifact_uri = tracker.log_artifacts(run_name, config=cfg, predictions=predictions)
    record = RunRecord(
        run_name=run_name,
        model=model_name,
        overrides={},
        metrics=metrics,
        metadata={"per_series": evaluation.per_series},
        artifact_uri=artifact_uri,
    )
    return BackendResult(record=record, predictions=predictions)


def endpoint_predict_step(
    cfg: "ExperimentConfig",
    endpoint_resource_name: str,
    *,
    aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
    frame_loader: "Callable[[ExperimentConfig], pd.DataFrame] | None" = None,
) -> "pd.DataFrame":
    """Online-predict every series against the endpoint; return the standardized frame."""
    aip = aiplatform or _lazy_aiplatform()
    _aip_init(aip, cfg)
    frame = (frame_loader or _default_frame_loader)(cfg)
    series_ids, contexts, last_dates = series_contexts(frame, cfg)
    instances = [
        {"series_id": sid, "context": ctx} for sid, ctx in zip(series_ids, contexts, strict=True)
    ]
    endpoint = aip.Endpoint(endpoint_resource_name)
    prediction = endpoint.predict(instances=instances)
    last_date_by_series = dict(zip(series_ids, last_dates, strict=True))
    return responses_to_frame(prediction.predictions, last_date_by_series, model_name="timesfm")


def batch_predict_timesfm_step(  # noqa: PLR0913 - cfg + name + injected seams (offline-testable)
    cfg: "ExperimentConfig",
    model_resource_name: str,
    *,
    aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
    frame_loader: "Callable[[ExperimentConfig], pd.DataFrame] | None" = None,
    instances_writer: "Callable[[ExperimentConfig, list[dict[str, Any]]], str] | None" = None,
    predictions_reader: "Callable[[Any], list[dict[str, Any]]] | None" = None,
) -> "pd.DataFrame":
    """Batch-predict every series with the custom container; return the standardized frame."""
    aip = aiplatform or _lazy_aiplatform()
    _aip_init(aip, cfg)
    serving = cfg.pipeline.serving
    frame = (frame_loader or _default_frame_loader)(cfg)
    series_ids, contexts, last_dates = series_contexts(frame, cfg)
    instances = [
        {"series_id": sid, "context": ctx} for sid, ctx in zip(series_ids, contexts, strict=True)
    ]
    gcs_source = (instances_writer or _default_instances_writer)(cfg, instances)
    model = aip.Model(model_resource_name)
    job = model.batch_predict(
        job_display_name=f"{timesfm_model_display_name(cfg)}__batch",
        gcs_source=gcs_source,
        gcs_destination_prefix=f"{pipeline_root(cfg)}/timesfm-batch",
        instances_format="jsonl",
        predictions_format="jsonl",
        machine_type=serving.batch_machine_type,
        starting_replica_count=serving.batch_starting_replica_count,
        max_replica_count=serving.batch_max_replica_count,
        labels=resource_labels(),
    )
    responses = (predictions_reader or _default_predictions_reader)(job)
    last_date_by_series = dict(zip(series_ids, last_dates, strict=True))
    return responses_to_frame(responses, last_date_by_series, model_name="timesfm")


def _default_instances_writer(  # pragma: no cover - live GCS IO
    cfg: "ExperimentConfig", instances: list[dict[str, Any]]
) -> str:
    """Write instances as JSONL to GCS and return the gs:// source URI."""
    import json  # noqa: PLC0415 - stdlib, lazy for symmetry

    from google.cloud import storage  # noqa: PLC0415 - lazy cloud import

    uri = f"{pipeline_root(cfg)}/timesfm-batch/instances.jsonl"
    bucket_name, _, blob_path = uri[len("gs://") :].partition("/")
    payload = "\n".join(json.dumps(inst) for inst in instances)
    storage.Client(project=cfg.project.id).bucket(bucket_name).blob(blob_path).upload_from_string(
        payload
    )
    return uri


def _default_predictions_reader(job: Any) -> list[dict[str, Any]]:  # noqa: ANN401  # pragma: no cover
    """Read the batch job's JSONL output and return the prediction response dicts."""
    msg = "reading custom-container batch output is resolved on the live run (Stage 4.9)"
    raise NotImplementedError(msg)


def teardown_serving_step(
    cfg: "ExperimentConfig",
    *,
    delete_model: bool = True,
    aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
) -> None:
    """Undeploy + delete the TimesFM endpoint(s) and model(s), resolved by display name.

    This is the transient-teardown step run as the pipeline's ``dsl.ExitHandler`` exit task, so it
    must work from config alone (the exit task can't reference the endpoint resource name produced
    inside the handler). It is best-effort and idempotent: if nothing matches (batch mode, TimesFM
    disabled, or already torn down) it is a no-op, so it is safe to always run on pipeline exit.
    """
    aip = aiplatform or _lazy_aiplatform()
    _aip_init(aip, cfg)
    endpoint_name = timesfm_endpoint_display_name(cfg)
    for endpoint in aip.Endpoint.list(filter=f'display_name="{endpoint_name}"'):
        logger.info("tearing down TimesFM endpoint %s", endpoint.resource_name)
        endpoint.undeploy_all()
        endpoint.delete()
    if delete_model:
        model_name = timesfm_model_display_name(cfg)
        for model in aip.Model.list(filter=f'display_name="{model_name}"'):
            logger.info("deleting TimesFM model %s", model.resource_name)
            model.delete()


def compare_step(results: "list[tuple[str, dict[str, float]]]") -> Comparison:
    """Rank backends by RMSE (tie-break MAE); the lowest-error model wins.

    Thin wrapper over :func:`~geaptimes.experiment.comparison.rank_backends` so the pipeline's
    ``compare-backends`` step and the ``run_experiment`` CLI rank backends identically.
    """
    comparison = rank_backends(results)
    logger.info("comparison winner: %s", comparison.winner)
    return comparison
