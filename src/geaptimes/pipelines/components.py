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
JSON parameters and collected by ``compare_backends`` to pick the winner.
"""

from dataclasses import dataclass
from typing import Any

from kfp import dsl


@dataclass(frozen=True)
class PipelineComponents:
    """The compiled KFP components for the comparison pipeline, bound to one runtime image."""

    build_tables: Any
    train_backend: Any
    infer_backend: Any
    score_and_track: Any
    register_timesfm: Any
    deploy_endpoint: Any
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
    def build_tables(config_json: str) -> str:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        names = steps.build_tables_step(cfg)
        return names["infer"]

    @dsl.component(base_image=image)
    def train_backend(config_json: str, model_name: str, tables: str) -> str:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        _ = tables  # ordering dependency on build_tables; the table names come from config
        cfg = ExperimentConfig.model_validate_json(config_json)
        return steps.train_backend_step(cfg, model_name)

    @dsl.component(base_image=image)
    def infer_backend(
        config_json: str,
        model_name: str,
        model_reference: str,
        predictions: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = steps.infer_backend_step(cfg, model_name, model_reference)
        frame.to_parquet(predictions.path, index=False)

    @dsl.component(base_image=image)
    def score_and_track(
        config_json: str,
        model_name: str,
        predictions: dsl.Input[dsl.Dataset],
    ) -> dict:
        import pandas as pd

        from geaptimes.experiment.tracking import ExperimentTracker
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = pd.read_parquet(predictions.path)
        result = steps.score_and_track_step(cfg, model_name, frame, tracker=ExperimentTracker(cfg))
        metrics = result.record.metrics
        return {
            "model": result.record.model,
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
        }

    @dsl.component(base_image=image)
    def register_timesfm(config_json: str) -> str:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        return steps.register_timesfm_step(cfg)

    @dsl.component(base_image=image)
    def deploy_endpoint(config_json: str, model_resource_name: str) -> str:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        return steps.deploy_endpoint_step(cfg, model_resource_name)

    @dsl.component(base_image=image)
    def endpoint_predict(
        config_json: str,
        endpoint_resource_name: str,
        predictions: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = steps.endpoint_predict_step(cfg, endpoint_resource_name)
        frame.to_parquet(predictions.path, index=False)

    @dsl.component(base_image=image)
    def batch_predict(
        config_json: str,
        model_resource_name: str,
        predictions: dsl.Output[dsl.Dataset],
    ) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        frame = steps.batch_predict_timesfm_step(cfg, model_resource_name)
        frame.to_parquet(predictions.path, index=False)

    @dsl.component(base_image=image)
    def teardown_serving(config_json: str) -> None:
        from geaptimes.pipelines import steps
        from geaptimes.schemas import ExperimentConfig

        cfg = ExperimentConfig.model_validate_json(config_json)
        steps.teardown_serving_step(cfg)

    @dsl.component(base_image=image)
    def compare_backends(rows: list, comparison: dsl.Output[dsl.Dataset]) -> str:
        import json
        from pathlib import Path

        from geaptimes.pipelines import steps

        result = steps.compare_step([(row["model"], row) for row in rows])
        with Path(comparison.path).open("w", encoding="utf-8") as fh:
            json.dump({"winner": result.winner, "ranking": result.ranking}, fh, indent=2)
        return result.winner

    return PipelineComponents(
        build_tables=build_tables,
        train_backend=train_backend,
        infer_backend=infer_backend,
        score_and_track=score_and_track,
        register_timesfm=register_timesfm,
        deploy_endpoint=deploy_endpoint,
        endpoint_predict=endpoint_predict,
        batch_predict=batch_predict,
        teardown_serving=teardown_serving,
        compare_backends=compare_backends,
    )
