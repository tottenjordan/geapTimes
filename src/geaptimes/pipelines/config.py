"""Pure derivations off :class:`~geaptimes.schemas.ExperimentConfig` for the pipeline layer.

These helpers centralize the cloud-resource names/URIs the pipeline needs (the runtime image URI,
the pipeline root, and the TimesFM model/endpoint display names) so the step functions and the KFP
components don't recompute them. All functions are pure and unit-tested offline.
"""

import hashlib
import json
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

# Label keys for warm-endpoint reuse. ``fingerprint`` makes a deployed model/endpoint discoverable
# by its serving contract; ``keep`` marks a warm (kept-running) deployment so the transient teardown
# never deletes it. GCP label values are ``[a-z0-9_-]{0,63}`` -- the fingerprint is truncated hex.
_FINGERPRINT_LABEL = "fingerprint"
_KEEP_LABEL = "keep"
_FINGERPRINT_LEN = 16

# Custom component executor sizing: machine type -> (cpu limit, memory limit). KFP lightweight
# components expose no machine-type knob; on Vertex Pipelines the CPU/memory limits select the
# executor machine, so we translate the configured machine type to those limits.
_COMPONENT_RESOURCES = {
    "e2-standard-2": ("2", "8G"),
    "e2-standard-4": ("4", "16G"),
    "e2-standard-8": ("8", "32G"),
}


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


def timesfm_artifact_uri(cfg: "ExperimentConfig") -> str:
    """GCS prefix recorded as the TimesFM model's ``artifact_uri``.

    The checkpoint is baked into the serving image, so there are no external model artifacts — but
    the GCPC ``importer`` node still requires a **non-empty** URI (an empty string fails at runtime
    with "Failed to get the URI of the artifact to import"). Point it at a stable, empty prefix
    under the project bucket; at deploy Vertex copies that prefix's (empty) contents to
    ``AIP_STORAGE_URI``, which the serving container ignores because it loads the baked checkpoint.
    """
    return f"gs://{cfg.project.gcs_bucket}/serving/{timesfm_model_display_name(cfg)}/"


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


def serving_fingerprint(cfg: "ExperimentConfig", image_digest: str) -> str:
    """Stable short hash identifying an interchangeable TimesFM deployment.

    Two TimesFM deployments are reuse-compatible iff their serving contract matches: the resolved
    image **digest** (the checkpoint is baked into the image, so the digest -- not the mutable
    ``:latest`` tag -- pins the model) plus the serving env vars (:func:`serving_env_vars`, which
    include checkpoint / max-context / context-len / horizon / quantiles). Hashing the digest is the
    point: a rebuilt image gets a new fingerprint, so a stale endpoint is never reused. Truncated to
    16 hex chars -- lowercase hex is GCP-label-safe and well under the 63-char label-value limit.
    """
    payload = json.dumps(
        {"image_digest": image_digest, "serving_env": serving_env_vars(cfg)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_FINGERPRINT_LEN]


def serving_labels(cfg: "ExperimentConfig", fingerprint: str = "") -> dict[str, str]:
    """Resource labels stamped on every deployed TimesFM model/endpoint.

    Always carries ``keep`` (``"true"`` when ``serving.keep_deployed`` -- a warm deployment the
    transient teardown's guard must skip). The ``keep`` label is independent of reuse: a warm
    endpoint created with ``keep_deployed=True`` but ``reuse_endpoint=False`` is still protected
    from a later transient run that shares its (config-derived) display name. The ``fingerprint``
    reuse key (:func:`serving_fingerprint`) is added only when one was resolved (reuse mode), so a
    deploy is discoverable for reuse without polluting non-reuse deployments with an empty label.
    """
    labels = {
        **resource_labels(),
        _KEEP_LABEL: "true" if cfg.pipeline.serving.keep_deployed else "false",
    }
    if fingerprint:
        labels[_FINGERPRINT_LABEL] = fingerprint
    return labels


def reusable_endpoint_filter(fingerprint: str) -> str:
    """Vertex ``Endpoint.list`` filter matching a warm endpoint by fingerprint + solution label."""
    solution = RESOURCE_LABELS["solution"]
    return f'labels.{_FINGERPRINT_LABEL}="{fingerprint}" AND labels.solution="{solution}"'


def is_kept_warm(labels: "dict[str, str] | None") -> bool:
    """Whether a resource's labels mark it as a kept-warm deployment (teardown must skip it)."""
    return bool(labels) and labels.get(_KEEP_LABEL) == "true"


def component_resources(cfg: "ExperimentConfig") -> tuple[str, str]:
    """``(cpu_limit, memory_limit)`` for the custom component pods, from ``component_machine_type``.

    Translating the machine type to KFP CPU/memory limits is what makes the right-size real: on
    Vertex Pipelines those limits select the smallest machine that fits (so ``e2-standard-2`` ->
    2 vCPU / 8 GB). An unsupported machine type fails loudly at compile time rather than silently
    falling back to a default size.
    """
    machine = cfg.pipeline.component_machine_type
    try:
        return _COMPONENT_RESOURCES[machine]
    except KeyError:
        supported = ", ".join(sorted(_COMPONENT_RESOURCES))
        msg = f"unsupported component_machine_type {machine!r}; supported: {supported}"
        raise ValueError(msg) from None


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
