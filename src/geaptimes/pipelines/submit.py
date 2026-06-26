"""Compile + submit the comparison pipeline as a Vertex AI ``PipelineJob``.

``submit_pipeline`` is the live entry point: it compiles the DAG for a config (pure), initializes
``aiplatform``, and submits a ``PipelineJob`` under the experiment so each backend lands as an
``ExperimentRun`` alongside the Stage 3 runs. The ``aiplatform`` module is injected, so the
compile + submit wiring unit-tests offline with a fake.

``with_automl_enabled`` is a DOE-style override (``model_dump`` → patch → ``model_validate``, never
in-place mutation): it flips on the AutoML backend for a one-off comparison without editing YAML —
exposed as ``--enable-automl`` on the CLI. The CLI's ``--dry-run`` compiles only (no submit, no
spend).
"""

import argparse
import tempfile
from pathlib import Path
from typing import Any

from geaptimes.pipelines.compile import compile_pipeline
from geaptimes.pipelines.config import (
    pipeline_job_display_name,
    pipeline_root,
    resource_labels,
)
from geaptimes.schemas import ExperimentConfig
from geaptimes.utils.logger import get_logger

logger = get_logger(__name__)


def with_automl_enabled(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a re-validated copy of *cfg* with the AutoML backend enabled.

    Flips ``enabled: true`` on the existing AutoML model, or appends a default AutoML model if the
    config has none. The input config is never mutated.
    """
    data = cfg.model_dump()
    models = data["models"]
    for model in models:
        if model["params"]["type"] == "automl":
            model["enabled"] = True
            break
    else:
        models.append({"name": "automl", "enabled": True, "params": {"type": "automl"}})
    return ExperimentConfig.model_validate(data)


def _lazy_aiplatform() -> Any:  # noqa: ANN401 - external SDK module
    from google.cloud import aiplatform  # noqa: PLC0415 - lazy cloud import

    return aiplatform


def _default_template_path() -> str:
    return str(Path(tempfile.mkdtemp(prefix="geaptimes-pipeline-")) / "comparison_pipeline.yaml")


def submit_pipeline(
    cfg: ExperimentConfig,
    *,
    aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
    template_path: "str | Path | None" = None,
    enable_caching: bool | None = None,
) -> Any:  # noqa: ANN401 - returns the SDK PipelineJob
    """Compile the comparison pipeline for *cfg* and submit it as a Vertex ``PipelineJob``."""
    aip = aiplatform or _lazy_aiplatform()
    path = str(template_path) if template_path is not None else _default_template_path()
    compile_pipeline(cfg, path)
    aip.init(
        project=cfg.project.id,
        location=cfg.project.region,
        staging_bucket=f"gs://{cfg.project.gcs_bucket}",
    )
    caching = enable_caching if enable_caching is not None else cfg.pipeline.enable_caching
    job = aip.PipelineJob(
        display_name=pipeline_job_display_name(cfg),
        template_path=path,
        pipeline_root=pipeline_root(cfg),
        parameter_values={"config_json": cfg.model_dump_json()},
        enable_caching=caching,
        labels=resource_labels(),
    )
    job.submit(experiment=cfg.execution.experiment_name)
    logger.info("submitted comparison PipelineJob %s", pipeline_job_display_name(cfg))
    return job


def main(argv: "list[str] | None" = None) -> int:
    """CLI: compile (``--dry-run``) or submit the comparison pipeline for a config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--enable-automl",
        action="store_true",
        help="Enable the AutoML backend for this run (overrides the config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile the pipeline only; do not submit (no cloud calls, no spend).",
    )
    parser.add_argument(
        "--output", default=None, help="Where to write the compiled pipeline YAML (optional)."
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.enable_automl:
        cfg = with_automl_enabled(cfg)

    if args.dry_run:
        path = compile_pipeline(cfg, args.output or _default_template_path())
        logger.info("compiled comparison pipeline to %s (dry-run; not submitted)", path)
        return 0

    submit_pipeline(cfg, template_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
