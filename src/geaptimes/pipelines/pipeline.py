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

from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from kfp import dsl
from kfp.dsl import importer

from geaptimes.naming import config_slug
from geaptimes.pipelines.components import build_components
from geaptimes.pipelines.config import (
    component_resources,
    image_uri,
    pipeline_root,
    serving_labels,
    timesfm_artifact_uri,
    timesfm_container_spec,
    timesfm_endpoint_display_name,
    timesfm_model_display_name,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kfp.dsl import PipelineTask
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


def build_pipeline(  # noqa: PLR0915 - one task per DAG node
    cfg: "ExperimentConfig",
    *,
    reused_endpoint: str = "",
    serving_fingerprint: str = "",
) -> "BaseComponent":
    """Build the comparison pipeline component for *cfg* (ready for ``compiler.Compiler``).

    ``reused_endpoint`` / ``serving_fingerprint`` are **compile-time** inputs resolved live at
    submit time (see :func:`~geaptimes.pipelines.submit.submit_pipeline`), not pipeline parameters.
    A non-empty ``reused_endpoint`` selects the warm-endpoint *reuse* branch: skip register/deploy,
    predict against that endpoint, and emit no teardown (we did not create it). On the fresh path a
    non-empty ``serving_fingerprint`` stamps the reuse labels on the deployed model + endpoint so a
    later run can discover and reuse them.
    """
    comps = build_components(image_uri(cfg))
    root = pipeline_root(cfg)
    name = pipeline_name(cfg)
    default_config_json = cfg.model_dump_json()
    serving = cfg.pipeline.serving
    # Producer caching is an explicit per-task opt-in: True renders ``cachingOptions.enableCache``
    # in the IR, which survives serialization and caches on Vertex. The pipeline-level default is
    # always submitted off (see submit_pipeline), so the ephemeral serving lifecycle -- which sets
    # caching False (an empty cachingOptions that falls back to that off default) -- never caches.
    caching = cfg.pipeline.enable_caching
    in_process = _in_process_model_names(cfg)
    timesfm = _timesfm_enabled(cfg)
    # No teardown task is emitted when we reused a warm endpoint (we did not create it) -- the
    # primary teardown guard; teardown_serving_step's keep-label skip is the secondary one.
    needs_teardown = (
        timesfm and serving.mode == "endpoint" and not serving.keep_deployed and not reused_endpoint
    )
    cpu_limit, memory_limit = component_resources(cfg)

    def _size(task: "PipelineTask") -> None:
        """Right-size a custom component's executor pod; GCPC ops keep their managed sizing."""
        task.set_cpu_limit(cpu_limit)
        task.set_memory_limit(memory_limit)

    def _body(config_json: str) -> None:  # noqa: PLR0915 - one task per DAG node + sizing
        # Self-bootstrapping data prep: ensure source -> prepped exist (guarded by an
        # existence + config-fingerprint check inside the steps), then derive train/infer. The
        # prepped Dataset artifact flows from ensure-prepped into build-tables AND the TimesFM
        # serving branch, so every backend has a real data-lineage edge back to prepped.
        source = comps.ensure_source(config_json=config_json)
        source.set_caching_options(caching)
        source.set_display_name("ensure-source")
        _size(source)
        prepped = comps.ensure_prepped(config_json=config_json, source=source.outputs["source"])
        prepped.set_caching_options(caching)
        prepped.set_display_name("ensure-prepped")
        _size(prepped)
        tables = comps.build_tables(config_json=config_json, prepped=prepped.outputs["prepped"])
        tables.set_caching_options(caching)
        tables.set_display_name("build-tables")
        _size(tables)
        rows = []
        for model_name in in_process:
            # Each in-process backend is a train -> infer -> score chain. The trained model is
            # passed by reference from train to infer (so an infer-side failure re-uses the cached
            # trained model), and the predictions flow to the one shared score-and-track step. The
            # train/infer table Dataset artifacts are wired in for data lineage (and ordering).
            train = comps.train_backend(
                config_json=config_json,
                model_name=model_name,
                train=tables.outputs["train"],
            )
            train.set_caching_options(caching)
            train.set_display_name(f"train:{model_name}")
            _size(train)
            infer = comps.infer_backend(
                config_json=config_json,
                model_name=model_name,
                model_reference=train.output,
                infer=tables.outputs["infer"],
            )
            infer.set_caching_options(caching)
            infer.set_display_name(f"infer:{model_name}")
            _size(infer)
            score = comps.score_and_track(
                config_json=config_json,
                model_name=model_name,
                predictions=infer.outputs["predictions"],
            )
            score.set_caching_options(caching)
            score.set_display_name(f"score:{model_name}")
            _size(score)
            rows.append(score.outputs["Output"])
        if timesfm and reused_endpoint:
            # Warm-endpoint reuse branch (resolved at submit time): a fingerprint-matching endpoint
            # is already serving, so skip the whole register -> create -> deploy lifecycle and
            # predict straight against it. No teardown is emitted (needs_teardown is False), so the
            # shared warm endpoint survives this run.
            served = comps.endpoint_predict_by_name(
                config_json=config_json,
                endpoint_resource_name=reused_endpoint,
                prepped=prepped.outputs["prepped"],
            )
            served.set_display_name("timesfm:endpoint-predict-by-name")
            served.set_caching_options(False)  # depends on the live endpoint; never cache
            _size(served)
            score_timesfm = comps.score_and_track(
                config_json=config_json,
                model_name="timesfm",
                predictions=served.outputs["predictions"],
            )
            score_timesfm.set_caching_options(caching)
            score_timesfm.set_display_name("score:timesfm")
            _size(score_timesfm)
            rows.append(score_timesfm.outputs["Output"])
        elif timesfm:
            # Hybrid serving lifecycle via GCPC ops (standardized google.Vertex* artifacts +
            # ML-Metadata lineage + billing-label propagation). Our TimesFM is a *custom container*
            # with a baked checkpoint, so ModelUploadOp consumes an UnmanagedContainerModel artifact
            # carrying the containerSpec (built by an importer node) rather than serving_container_*
            # args. project/location/display names are baked at compile time (compiled per submit).
            # Always stamp the keep label (mirrors keep_deployed) so a warm endpoint is protected
            # from a later transient run's teardown even without reuse mode; add the fingerprint
            # label (discoverable for reuse) when one was resolved at submit time.
            labels = serving_labels(cfg, serving_fingerprint)
            container_spec = importer(
                # Empty GCS prefix: checkpoint is baked into the image, but the importer node
                # requires a non-empty URI (an empty string fails the import at runtime).
                artifact_uri=timesfm_artifact_uri(cfg),
                artifact_class=artifact_types.UnmanagedContainerModel,
                metadata={"containerSpec": timesfm_container_spec(cfg)},
            )
            # Deterministic producer (containerSpec fixed at compile): honor the producer-caching
            # flag like the others. A KFP importer defaults to cacheable, so without this it would
            # stay cached even under --no-cache.
            container_spec.set_caching_options(caching)
            container_spec.set_display_name("timesfm:container-spec")
            register = ModelUploadOp(
                project=cfg.project.id,
                location=cfg.project.region,
                display_name=timesfm_model_display_name(cfg),
                unmanaged_container_model=container_spec.outputs["artifact"],
                labels=labels,
            )
            register.set_caching_options(caching)
            register.set_display_name("register-timesfm")
            if serving.mode == "endpoint":
                endpoint = EndpointCreateOp(
                    project=cfg.project.id,
                    location=cfg.project.region,
                    display_name=timesfm_endpoint_display_name(cfg),
                    labels=labels,
                )
                # Never cache the ephemeral endpoint lifecycle: a cached endpoint-create hands back
                # an endpoint that a prior run's ExitHandler teardown already deleted, so the deploy
                # then targets a non-existent endpoint. The model registry tasks above stay
                # cacheable (their model artifacts survive teardown); only create/deploy/predict/
                # teardown must re-run every submit.
                endpoint.set_caching_options(False)
                endpoint.set_display_name("create-endpoint")
                deploy = ModelDeployOp(
                    model=register.outputs["model"],
                    endpoint=endpoint.outputs["endpoint"],
                    deployed_model_display_name=timesfm_endpoint_display_name(cfg),
                    dedicated_resources_machine_type=serving.machine_type,
                    dedicated_resources_min_replica_count=serving.min_replica_count,
                    dedicated_resources_max_replica_count=serving.max_replica_count,
                    traffic_split={"0": "100"},
                )
                deploy.set_caching_options(False)  # see create-endpoint: ephemeral, never cache
                deploy.set_display_name("deploy-endpoint")
                # endpoint_predict stays custom (online predict + our frame mapping). It consumes
                # the VertexEndpoint artifact and must run *after* the deploy completes.
                served = comps.endpoint_predict(
                    config_json=config_json,
                    endpoint=endpoint.outputs["endpoint"],
                    prepped=prepped.outputs["prepped"],
                )
                served.after(deploy)
                served.set_display_name("timesfm:endpoint-predict")
            else:
                served = comps.batch_predict(
                    config_json=config_json,
                    model=register.outputs["model"],
                    prepped=prepped.outputs["prepped"],
                )
                served.set_display_name("timesfm:batch-predict")
            served.set_caching_options(False)  # depends on the live deployment; never cache
            _size(served)
            # TimesFM has no training step; its served predictions feed the same shared scorer.
            score_timesfm = comps.score_and_track(
                config_json=config_json,
                model_name="timesfm",
                predictions=served.outputs["predictions"],
            )
            score_timesfm.set_caching_options(caching)
            score_timesfm.set_display_name("score:timesfm")
            _size(score_timesfm)
            rows.append(score_timesfm.outputs["Output"])
        compare = comps.compare_backends(rows=rows)
        compare.set_caching_options(caching)
        compare.set_display_name("compare-backends")
        _size(compare)

    @dsl.pipeline(name=name, pipeline_root=root)
    def comparison(config_json: str = default_config_json) -> None:
        if needs_teardown:
            teardown = comps.teardown_serving(config_json=config_json)
            teardown.set_caching_options(False)  # must always run to delete the transient endpoint
            teardown.set_display_name("teardown-serving")
            _size(teardown)
            with dsl.ExitHandler(teardown):
                _body(config_json)
        else:
            _body(config_json)

    return comparison
