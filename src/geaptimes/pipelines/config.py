"""Pure derivations off :class:`~geaptimes.schemas.ExperimentConfig` for the pipeline layer.

These helpers centralize the cloud-resource names/URIs the pipeline needs (the runtime image URI,
the pipeline root, and the TimesFM model/endpoint display names) so the step functions and the KFP
components don't recompute them. All functions are pure and unit-tested offline.
"""

from typing import TYPE_CHECKING

from geaptimes.constants import RESOURCE_LABELS
from geaptimes.naming import config_slug

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geaptimes.schemas import ExperimentConfig


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
