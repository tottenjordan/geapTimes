"""Pure derivations off :class:`~geaptimes.schemas.ExperimentConfig` for the pipeline layer.

These helpers centralize the cloud-resource names/URIs the pipeline needs (the runtime image URI,
the pipeline root, and the TimesFM model/endpoint display names) so the step functions and the KFP
components don't recompute them. All functions are pure and unit-tested offline.
"""

from typing import TYPE_CHECKING, Any

from geaptimes.constants import RESOURCE_LABELS
from geaptimes.naming import config_slug

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geaptimes.schemas import ExperimentConfig

# TimesFM serving-container defaults (used when no explicit timesfm model params are present).
_DEFAULT_CONTEXT_LEN = 512
_DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
_DEFAULT_MAX_CONTEXT = 1024
_HEALTH_ROUTE = "/health"
_PREDICT_ROUTE = "/predict"
_SERVING_PORT = 8080


def image_uri(cfg: "ExperimentConfig") -> str:
    """Fully-qualified Artifact Registry URI for the geapTimes runtime image."""
    return cfg.pipeline.image.image_uri(project_id=cfg.project.id, region=cfg.project.region)


def pipeline_root(cfg: "ExperimentConfig") -> str:
    """The ``gs://`` pipeline root for Vertex Pipelines artifacts."""
    return cfg.pipeline.resolved_pipeline_root(
        gcs_bucket=cfg.project.gcs_bucket,
        experiment_name=cfg.execution.experiment_name,
    )


def timesfm_model_display_name(cfg: "ExperimentConfig") -> str:
    """Display name for the registered TimesFM serving model."""
    return f"timesfm__{config_slug(cfg.data, cfg.forecast)}"


def timesfm_endpoint_display_name(cfg: "ExperimentConfig") -> str:
    """Display name for the TimesFM online endpoint."""
    return f"timesfm-endpoint__{config_slug(cfg.data, cfg.forecast)}"


def pipeline_job_display_name(cfg: "ExperimentConfig") -> str:
    """Display name for the comparison PipelineJob."""
    return f"geaptimes-comparison__{config_slug(cfg.data, cfg.forecast)}"


def resource_labels() -> dict[str, str]:
    """A fresh copy of the solution resource labels."""
    return dict(RESOURCE_LABELS)


def timesfm_context_len(cfg: "ExperimentConfig") -> int:
    """Context length from the timesfm model params (default if absent)."""
    for model_cfg in cfg.models:
        params = model_cfg.params
        if params.type == "timesfm":
            return params.context_len
    return _DEFAULT_CONTEXT_LEN


def serving_env_vars(cfg: "ExperimentConfig") -> dict[str, str]:
    """Environment variables the TimesFM serving container reads to compile the predictor."""
    checkpoint = _DEFAULT_CHECKPOINT
    max_context = _DEFAULT_MAX_CONTEXT
    quantiles = True
    for model_cfg in cfg.models:
        if model_cfg.params.type == "timesfm":
            checkpoint = model_cfg.params.checkpoint
            max_context = model_cfg.params.max_context
            quantiles = model_cfg.params.quantiles
            break
    return {
        "TIMESFM_CHECKPOINT": checkpoint,
        "TIMESFM_MAX_CONTEXT": str(max_context),
        "TIMESFM_CONTEXT_LEN": str(timesfm_context_len(cfg)),
        "TIMESFM_HORIZON": str(cfg.forecast.horizon),
        "TIMESFM_QUANTILES": "true" if quantiles else "false",
    }


def timesfm_container_spec(cfg: "ExperimentConfig") -> dict[str, Any]:
    """Vertex ``ModelContainerSpec`` (JSON) for the baked TimesFM serving image.

    Fed to the ``importer`` node as the ``UnmanagedContainerModel`` artifact's ``containerSpec`` so
    GCPC's ``ModelUploadOp`` can register our custom container. The checkpoint is baked into the
    image (``HF_HUB_OFFLINE``), so there is no model-artifact URI — only the container + its routes,
    port, and predictor env.
    """
    return {
        "imageUri": image_uri(cfg),
        "predictRoute": _PREDICT_ROUTE,
        "healthRoute": _HEALTH_ROUTE,
        "ports": [{"containerPort": _SERVING_PORT}],
        "env": [{"name": key, "value": value} for key, value in serving_env_vars(cfg).items()],
    }
