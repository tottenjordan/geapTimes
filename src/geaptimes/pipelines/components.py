"""KFP v2 component shells over the :mod:`geaptimes.pipelines.steps` functions.

Each component is a thin lightweight-Python wrapper that reconstructs the typed
:class:`~geaptimes.schemas.ExperimentConfig` from a JSON pipeline parameter and delegates to the
matching step function — the real logic lives in ``steps.py`` (and is unit-tested offline there).
The components are built by :func:`build_components`, which binds them to the geapTimes runtime
image (the same custom container used for TimesFM serving); KFP overrides the container entrypoint,
so the serving ``CMD`` is inert when the image runs as a component executor.

Component bodies are executed inside that container, so all imports are function-local and reference
only the installed ``geaptimes`` package. Predictions flow between components as Parquet
``Dataset`` artifacts on the pipeline root; per-backend ``{model, mae, rmse}`` dicts are returned as
JSON parameters and collected by ``compare_backends`` to pick the winner. For at-a-glance results in
the Vertex Pipelines UI, ``score_and_track`` also emits a ``Metrics`` artifact (per-backend MAE /
RMSE scalars) and ``compare_backends`` a ``Markdown`` ranking table.
"""

from dataclasses import dataclass
from typing import Any

from kfp import dsl


@dataclass(frozen=True)
class PipelineComponents:
    """The compiled KFP components for the comparison pipeline, bound to one runtime image."""

    ensure_source: Any
    ensure_prepped: Any
    build_tables: Any
    train_backend: Any
    infer_backend: Any
    score_and_track: Any
    endpoint_predict: Any
    batch_predict: Any
    teardown_serving: Any
    compare_backends: Any


def build_components(image: str) -> PipelineComponents:  # noqa: PLR0915, C901 - one nested def per component
    """Build the comparison-pipeline components bound to *image* (the geapTimes runtime container).

    ``base_image`` must be fixed when ``@dsl.component`` is applied, so the components are created
    here (per resolved image URI) rather than at module import — keeping the layer pure and letting
    the compile step derive the image from config.
    """

    @dsl.component(base_image=image)
    def ensure_source(config_json: str, source: dsl.Output[dsl.Dataset]) -> None:
        from geaptimes.naming import config_fingerprint_hash
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        ref = steps.ensure_source_step(cfg)
        # Emit a table-reference lineage Dataset artifact: a BigQuery *reference* (bq:// uri +
        # fingerprint + rows), not the data itself, opening the source->prepped lineage chain.
        source.uri = f"bq://{ref.name}"
        source.metadata["table"] = ref.name
        source.metadata["rows"] = ref.rows
        source.metadata["fingerprint"] = config_fingerprint_hash(cfg)

    @dsl.component(base_image=image)
    def ensure_prepped(
        config_json: str,
        source: dsl.Input[dsl.Dataset],
        prepped: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.naming import config_fingerprint_hash
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = source  # lineage + ordering edge: prepped is derived from the source table
        cfg = ExperimentConfig.model_validate_json(config_json)
        ref = steps.ensure_prepped_step(cfg)
        prepped.uri = f"bq://{ref.name}"
        prepped.metadata["table"] = ref.name
        prepped.metadata["rows"] = ref.rows
        prepped.metadata["fingerprint"] = config_fingerprint_hash(cfg)

    @dsl.component(base_image=image)
    def build_tables(
        config_json: str,
        prepped: dsl.Input[dsl.Dataset],
        train: dsl.Output[dsl.Dataset],
        infer: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.naming import config_fingerprint_hash
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = prepped  # lineage + ordering edge: train/infer are derived from the prepped table
        cfg = ExperimentConfig.model_validate_json(config_json)
        refs = steps.build_tables_step(cfg)
        fingerprint = config_fingerprint_hash(cfg)
        # One lineage Dataset artifact per built table (bq:// reference, not the data itself).
        for role, artifact in (("train", train), ("infer", infer)):
            ref = refs[role]
            artifact.uri = f"bq://{ref.name}"
            artifact.metadata["table"] = ref.name
            artifact.metadata["rows"] = ref.rows
            artifact.metadata["fingerprint"] = fingerprint

    @dsl.component(base_image=image)
    def train_backend(
        config_json: str,
        model_name: str,
        train: dsl.Input[dsl.Dataset],
    ) -> str:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = train  # lineage + ordering edge from build_tables; table names come from config
        cfg = ExperimentConfig.model_validate_json(config_json)
        return steps.train_backend_step(cfg, model_name)

    @dsl.component(base_image=image)
    def infer_backend(
        config_json: str,
        model_name: str,
        model_reference: str,
        infer: dsl.Input[dsl.Dataset],
        predictions: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = infer  # lineage + ordering edge from build_tables; table names come from config
        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = steps.infer_backend_step(cfg, model_name, model_reference)
        frame.to_parquet(predictions.path, index=False)

    @dsl.component(base_image=image)
    def score_and_track(
        config_json: str,
        model_name: str,
        predictions: dsl.Input[dsl.Dataset],
        metrics: dsl.Output[dsl.Metrics],
    ) -> str:
        import base64
        import json

        import pandas as pd

        from geaptimes.experiment.tracking import ExperimentTracker
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = pd.read_parquet(predictions.path)
        result = steps.score_and_track_step(cfg, model_name, frame, tracker=ExperimentTracker(cfg))
        scores = result.record.metrics
        mae = float(scores["mae"])
        rmse = float(scores["rmse"])
        # Scalar metrics on the task itself (the system.Metrics artifact KFP renders in the Vertex
        # Pipelines UI), in addition to the Vertex ExperimentRun logged inside the step.
        metrics.log_metric("mae", mae)
        metrics.log_metric("rmse", rmse)
        # Return base64(JSON), not a dict or raw JSON string. compare_backends collects these into a
        # ``list`` param, which KFP builds by *textual* placeholder substitution into the executor
        # input JSON -- a raw JSON value's quotes would break that JSON (and a dict has no
        # resolvable list-element type). base64's alphabet is both JSON- and substitution-safe.
        payload = json.dumps({"model": result.record.model, "mae": mae, "rmse": rmse})
        return base64.b64encode(payload.encode("utf-8")).decode("ascii")

    @dsl.component(base_image=image)
    def endpoint_predict(
        config_json: str,
        endpoint: dsl.Input[dsl.Artifact],
        prepped: dsl.Input[dsl.Dataset],
        predictions: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = prepped  # lineage edge: the online predictions read context from the prepped table
        # ``endpoint`` is the google.VertexEndpoint artifact from GCPC's EndpointCreateOp; the
        # resource name lives in its metadata (consumed as a base Artifact so we don't depend on
        # the GCPC artifact types inside the container).
        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = steps.endpoint_predict_step(cfg, endpoint.metadata["resourceName"])
        frame.to_parquet(predictions.path, index=False)

    @dsl.component(base_image=image)
    def batch_predict(
        config_json: str,
        model: dsl.Input[dsl.Artifact],
        prepped: dsl.Input[dsl.Dataset],
        predictions: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = prepped  # lineage edge: the batch predictions read context from the prepped table
        # ``model`` is the google.VertexModel artifact from GCPC's ModelUploadOp (resource name in
        # its metadata).
        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = steps.batch_predict_timesfm_step(cfg, model.metadata["resourceName"])
        frame.to_parquet(predictions.path, index=False)

    @dsl.component(base_image=image)
    def teardown_serving(config_json: str) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        steps.teardown_serving_step(cfg)

    @dsl.component(base_image=image)
    def compare_backends(
        rows: list,
        comparison: dsl.Output[dsl.Dataset],
        ranking_md: dsl.Output[dsl.Markdown],
    ) -> str:
        import base64
        import json
        from pathlib import Path

        from geaptimes.pipelines import steps

        # ``rows`` are base64(JSON) strings emitted by score_and_track (see the note there).
        parsed = [json.loads(base64.b64decode(row)) for row in rows]
        result = steps.compare_step([(row["model"], row) for row in parsed])
        with Path(comparison.path).open("w", encoding="utf-8") as fh:
            json.dump({"winner": result.winner, "ranking": result.ranking}, fh, indent=2)
        # Human-readable ranking table (the system.Markdown artifact rendered in the Vertex
        # Pipelines UI); the winner row is flagged so the outcome is legible at a glance.
        lines = ["| rank | model | mae | rmse |", "| --- | --- | --- | --- |"]
        for idx, row in enumerate(result.ranking, start=1):
            flag = " (winner)" if row["model"] == result.winner else ""
            lines.append(f"| {idx} | {row['model']}{flag} | {row['mae']:.4f} | {row['rmse']:.4f} |")
        Path(ranking_md.path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result.winner

    return PipelineComponents(
        ensure_source=ensure_source,
        ensure_prepped=ensure_prepped,
        build_tables=build_tables,
        train_backend=train_backend,
        infer_backend=infer_backend,
        score_and_track=score_and_track,
        endpoint_predict=endpoint_predict,
        batch_predict=batch_predict,
        teardown_serving=teardown_serving,
        compare_backends=compare_backends,
    )
