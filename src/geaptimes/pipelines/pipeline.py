"""The side-by-side comparison pipeline (KFP v2 DAG).

``build_pipeline(cfg)`` assembles the comparison DAG for a given config: build the train/infer
tables, run every enabled in-process backend (BQML, AutoML), serve TimesFM via the custom container
(online endpoint *or* batch prediction), then compare all backends and pick the winner. Each backend
is one Vertex ``ExperimentRun`` (handled inside the step functions) and writes its predictions as a
Parquet ``Dataset`` artifact on the pipeline root.

Pipeline *structure* is resolved at compile time from the typed config — which backends are enabled,
the serving ``mode``, and whether the endpoint is torn down (``keep_deployed``) — because the config
is known when we compile (we compile per submit). The rich config travels to the component bodies as
a single JSON pipeline parameter (``config_json``), so the same compiled graph stays self-contained
yet overridable at submit. Transient teardown uses a ``dsl.ExitHandler`` wrapping the whole body, so
the endpoint is never stranded on partial failure; the teardown step is a no-op when there is
nothing deployed.
"""

from typing import TYPE_CHECKING

from kfp import dsl

from geaptimes.naming import config_slug
from geaptimes.pipelines.components import build_components
from geaptimes.pipelines.config import image_uri, pipeline_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kfp.dsl.base_component import BaseComponent

    from geaptimes.schemas import ExperimentConfig

_IN_PROCESS_BACKENDS = ("bqml", "automl")


def _in_process_model_names(cfg: "ExperimentConfig") -> list[str]:
    """Enabled backends that run in-process (BQML / AutoML); TimesFM is served separately."""
    return [m.name for m in cfg.models if m.enabled and m.params.type in _IN_PROCESS_BACKENDS]


def _timesfm_enabled(cfg: "ExperimentConfig") -> bool:
    """Whether an enabled TimesFM model should be served + compared."""
    return any(m.enabled and m.params.type == "timesfm" for m in cfg.models)


def pipeline_name(cfg: "ExperimentConfig") -> str:
    """KFP pipeline spec name (hyphenated, lowercase — no underscores)."""
    return f"geaptimes-comparison-{config_slug(cfg.data, cfg.forecast).replace('_', '-')}"


def build_pipeline(cfg: "ExperimentConfig") -> "BaseComponent":
    """Build the comparison pipeline component for *cfg* (ready for ``compiler.Compiler``)."""
    comps = build_components(image_uri(cfg))
    root = pipeline_root(cfg)
    name = pipeline_name(cfg)
    default_config_json = cfg.model_dump_json()
    serving = cfg.pipeline.serving
    caching = cfg.pipeline.enable_caching
    in_process = _in_process_model_names(cfg)
    timesfm = _timesfm_enabled(cfg)
    needs_teardown = timesfm and serving.mode == "endpoint" and not serving.keep_deployed

    def _body(config_json: str) -> None:
        tables = comps.build_tables(config_json=config_json)
        tables.set_caching_options(caching)
        rows = []
        for model_name in in_process:
            backend = comps.run_backend(
                config_json=config_json, model_name=model_name, tables=tables.output
            )
            backend.set_caching_options(caching)
            rows.append(backend.outputs["Output"])
        if timesfm:
            register = comps.register_timesfm(config_json=config_json)
            register.set_caching_options(caching)
            if serving.mode == "endpoint":
                deploy = comps.deploy_endpoint(
                    config_json=config_json, model_resource_name=register.output
                )
                deploy.set_caching_options(caching)
                served = comps.endpoint_predict(
                    config_json=config_json, endpoint_resource_name=deploy.output
                )
            else:
                served = comps.batch_predict(
                    config_json=config_json, model_resource_name=register.output
                )
            served.set_caching_options(caching)
            rows.append(served.outputs["Output"])
        compare = comps.compare_backends(rows=rows)
        compare.set_caching_options(caching)

    @dsl.pipeline(name=name, pipeline_root=root)
    def comparison(config_json: str = default_config_json) -> None:
        if needs_teardown:
            with dsl.ExitHandler(comps.teardown_serving(config_json=config_json)):
                _body(config_json)
        else:
            _body(config_json)

    return comparison
