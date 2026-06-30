"""Compile + submit the comparison pipeline as a Vertex AI ``PipelineJob``.

``submit_pipeline`` is the live entry point: it compiles the DAG for a config (pure), initializes
``aiplatform``, and submits a ``PipelineJob`` under the experiment so each backend lands as an
``ExperimentRun`` alongside the Stage 3 runs. The ``aiplatform`` module is injected, so the
compile + submit wiring unit-tests offline with a fake.

``with_automl_enabled`` / ``with_automl_disabled`` are DOE-style overrides (``model_dump`` → patch →
``model_validate``, never in-place mutation): they flip the AutoML backend on/off for a one-off run
without editing YAML — exposed as the mutually-exclusive ``--enable-automl`` / ``--disable-automl``
CLI flags. ``--disable-automl`` is the cheap pipeline-test switch: it skips the long, billable
AutoML training so the rest of the DAG can be iterated in minutes. The CLI's ``--dry-run`` compiles
only (no submit, no spend).
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


def with_automl_disabled(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a re-validated copy of *cfg* with the AutoML backend disabled.

    The mirror of :func:`with_automl_enabled`: flips ``enabled: false`` on every AutoML model so a
    cheap pipeline test run skips the long, billable AutoML training (the rest of the DAG finishes
    in minutes). A config with no AutoML model is returned unchanged. The input config is never
    mutated.
    """
    data = cfg.model_dump()
    for model in data["models"]:
        if model["params"]["type"] == "automl":
            model["enabled"] = False
    return ExperimentConfig.model_validate(data)


def with_data_rebuild_forced(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a re-validated copy of *cfg* with the data-prep rebuild guard bypassed.

    Sets ``data.force_rebuild: true`` so ``ensure_source`` / ``ensure_prepped`` rebuild the source
    and prepped tables even when the stored fingerprint is current -- the manual override for when
    the upstream public data changed but the config (hence the fingerprint) did not. Exposed as the
    ``--force-data-rebuild`` CLI flag. The input config is never mutated.
    """
    data = cfg.model_dump()
    data["data"]["force_rebuild"] = True
    return ExperimentConfig.model_validate(data)


def with_caching_disabled(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a re-validated copy of *cfg* with deterministic-producer caching turned off.

    Sets ``pipeline.enable_caching: false`` so the producer tasks (data-prep, build-tables,
    train/infer, model-upload, ...) compile with no explicit ``set_caching_options(True)`` and thus
    fall back to the always-off Vertex pipeline default -- a full from-scratch re-run. The ephemeral
    serving lifecycle is never cached either way. Exposed as the ``--no-cache`` CLI flag. The input
    config is never mutated.
    """
    data = cfg.model_dump()
    data["pipeline"]["enable_caching"] = False
    return ExperimentConfig.model_validate(data)


def _lazy_aiplatform() -> Any:  # noqa: ANN401 - external SDK module
    from google.cloud import aiplatform  # noqa: PLC0415 - lazy cloud import

    return aiplatform


def _cleanup_preflight(forecaster: Any) -> None:  # noqa: ANN401 - holds SDK handles
    """Best-effort teardown of the probe resources a preflight ``fit()`` may have created.

    On a *valid* config the training pipeline is really created and starts training, so we cancel
    it (stops the billable compute) and then delete it (cancel leaves a CANCELLED record behind).
    On an *invalid* config no pipeline resource exists, so these are no-ops. The probe dataset is
    always removed. Every action is best-effort -- cleanup must never mask the real result.
    """
    job = forecaster.training_job
    if job is not None:
        for action in ("cancel", "delete"):
            try:
                getattr(job, action)()
            except Exception:  # noqa: BLE001 - cleanup must not mask the real result
                logger.warning("preflight: training-job %s failed", action, exc_info=True)
    dataset = forecaster.dataset
    if dataset is not None:
        try:
            dataset.delete()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real result
            logger.warning("preflight: could not delete probe dataset", exc_info=True)


def preflight_automl(
    cfg: ExperimentConfig,
    *,
    aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
) -> None:
    """Validate the AutoML backend against Vertex's real rules WITHOUT a full pipeline deploy.

    The role/transformation errors all come from Vertex's server-side ``create_training_pipeline``
    validation, which the deployed pipeline only reaches after a ~4-minute image-build + submit +
    pod-startup cycle. This runs the same AutoML ``fit()`` locally with ``sync=False`` so that
    validation executes in seconds; on a valid config the launched training pipeline is immediately
    cancelled and the probe dataset deleted (no training spend). An invalid config raises with
    Vertex's own message. Run this before ``submit_pipeline`` to fail fast and cheap.
    """
    from geaptimes.models.automl import AutoMLForecaster  # noqa: PLC0415 - keep model dep lazy

    aip = aiplatform or _lazy_aiplatform()
    enabled = with_automl_enabled(cfg)
    model = next(m for m in enabled.models if m.params.type == "automl")
    forecaster = AutoMLForecaster(model, enabled, aiplatform=aip, sync=False)
    logger.info("AutoML preflight: static invariant check")
    forecaster.validate_config()
    logger.info("AutoML preflight: running Vertex server-side validation")
    try:
        forecaster.fit()  # sync=False: schedules the create on a background thread
        # Block only until the training pipeline RESOURCE is created (seconds) -- this is where
        # Vertex's column/role validation runs and surfaces InvalidArgument -- not until training
        # completes. A valid config means a real pipeline now exists; cleanup cancels it.
        forecaster.training_job.wait_for_resource_creation()
    except Exception:
        logger.exception("AutoML preflight FAILED")
        raise
    else:
        logger.info("AutoML preflight PASSED: Vertex accepted the config")
    finally:
        _cleanup_preflight(forecaster)


def _default_template_path() -> str:
    return str(Path(tempfile.mkdtemp(prefix="geaptimes-pipeline-")) / "comparison_pipeline.yaml")


def submit_pipeline(
    cfg: ExperimentConfig,
    *,
    aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
    template_path: "str | Path | None" = None,
) -> Any:  # noqa: ANN401 - returns the SDK PipelineJob
    """Compile the comparison pipeline for *cfg* and submit it as a Vertex ``PipelineJob``.

    Caching is a *compile-time, per-task* concern, not a submit knob: deterministic producers opt
    in via an explicit ``set_caching_options(True)`` (driven by ``cfg.pipeline.enable_caching``;
    turn it off with ``--no-cache`` / :func:`with_caching_disabled`). The Vertex *pipeline-level*
    ``enable_caching`` is therefore always submitted ``False`` -- a task whose
    ``set_caching_options(False)`` serializes to an empty ``cachingOptions`` (KFP omits false
    booleans) must fall back to an off default, or the ephemeral endpoint-create caches and hands
    the deploy a torn-down endpoint. Explicit ``True`` on producers survives and still caches.
    """
    aip = aiplatform or _lazy_aiplatform()
    path = str(template_path) if template_path is not None else _default_template_path()
    compile_pipeline(cfg, path)
    aip.init(
        project=cfg.project.id,
        location=cfg.project.region,
        staging_bucket=f"gs://{cfg.project.gcs_bucket}",
    )
    job = aip.PipelineJob(
        display_name=pipeline_job_display_name(cfg),
        template_path=path,
        pipeline_root=pipeline_root(cfg),
        parameter_values={"config_json": cfg.model_dump_json()},
        enable_caching=False,  # always off at pipeline level; producers opt back in per-task
        labels=resource_labels(),
    )
    job.submit(experiment=cfg.execution.experiment_name)
    logger.info("submitted comparison PipelineJob %s", pipeline_job_display_name(cfg))
    return job


def main(argv: "list[str] | None" = None) -> int:
    """CLI: compile (``--dry-run``) or submit the comparison pipeline for a config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    automl_group = parser.add_mutually_exclusive_group()
    automl_group.add_argument(
        "--enable-automl",
        action="store_true",
        help="Enable the AutoML backend for this run (overrides the config).",
    )
    automl_group.add_argument(
        "--disable-automl",
        action="store_true",
        help="Disable the AutoML backend for this run (overrides the config) to skip the long, "
        "billable AutoML training while iterating on the pipeline.",
    )
    parser.add_argument(
        "--force-data-rebuild",
        action="store_true",
        help="Rebuild the source/prepped tables even if current (bypass the fingerprint guard).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable deterministic-producer caching for this run (full from-scratch re-run); the "
        "ephemeral serving lifecycle is never cached regardless.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile the pipeline only; do not submit (no cloud calls, no spend).",
    )
    parser.add_argument(
        "--preflight-automl",
        action="store_true",
        help="Validate the AutoML config against Vertex live (seconds, no training spend); "
        "do not submit the pipeline.",
    )
    parser.add_argument(
        "--output", default=None, help="Where to write the compiled pipeline YAML (optional)."
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.enable_automl:
        cfg = with_automl_enabled(cfg)
    elif args.disable_automl:
        cfg = with_automl_disabled(cfg)
    if args.force_data_rebuild:
        cfg = with_data_rebuild_forced(cfg)
    if args.no_cache:
        cfg = with_caching_disabled(cfg)

    if args.preflight_automl:
        preflight_automl(cfg)
        logger.info("AutoML preflight complete; pipeline not submitted")
        return 0

    if args.dry_run:
        path = compile_pipeline(cfg, args.output or _default_template_path())
        logger.info("compiled comparison pipeline to %s (dry-run; not submitted)", path)
        return 0

    submit_pipeline(cfg, template_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
