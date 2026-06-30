"""Provision the GCS bucket and BigQuery dataset for a geapTimes experiment config.

Idempotent: existing resources are left untouched. ``--dry-run`` previews actions without
contacting GCP (and without needing credentials).
"""

import argparse
from typing import TYPE_CHECKING

from google.cloud import bigquery, storage

from geaptimes.constants import RESOURCE_LABELS, SOLUTION
from geaptimes.schemas import ExperimentConfig, PipelineConfig, ProjectConfig
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = get_logger("geaptimes.setup_gcp")


def ensure_dataset(project: ProjectConfig, *, dry_run: bool) -> None:
    """Create the BigQuery dataset if it does not already exist."""
    dataset_id = f"{project.id}.{project.bq_dataset}"
    if dry_run:
        logger.info(
            "[dry-run] would ensure BigQuery dataset %s (%s)", dataset_id, project.bq_location
        )
        return
    client = bigquery.Client(project=project.id)
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = project.bq_location
    dataset.labels = RESOURCE_LABELS
    client.create_dataset(dataset, exists_ok=True)
    logger.info("Ensured BigQuery dataset %s", dataset_id)


def ensure_bucket(project: ProjectConfig, *, dry_run: bool) -> None:
    """Create the GCS bucket if it does not already exist."""
    if dry_run:
        logger.info(
            "[dry-run] would ensure GCS bucket gs://%s (%s)", project.gcs_bucket, project.region
        )
        return
    client = storage.Client(project=project.id)
    bucket = client.bucket(project.gcs_bucket)
    if bucket.exists():
        logger.info("GCS bucket gs://%s already exists", project.gcs_bucket)
        return
    bucket.labels = RESOURCE_LABELS
    client.create_bucket(bucket, location=project.region)
    logger.info("Created GCS bucket gs://%s", project.gcs_bucket)


def _run_command(cmd: list[str]) -> tuple[int, str, str]:  # pragma: no cover - shells out to gcloud
    """Run a command, returning ``(returncode, stdout, stderr)`` — the default gcloud runner."""
    import subprocess  # noqa: PLC0415 - stdlib, lazy (only the live path shells out)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.returncode, proc.stdout, proc.stderr


def ensure_artifact_registry(
    project: ProjectConfig,
    pipeline: PipelineConfig,
    *,
    dry_run: bool,
    runner: "Callable[[list[str]], tuple[int, str, str]] | None" = None,
) -> None:
    """Create the labelled Docker Artifact Registry repo for the runtime image (idempotent).

    Uses ``gcloud`` (no extra SDK dependency in the runtime image) and treats an already-existing
    repo as success. The repo holds the geapTimes runtime container built by ``cloudbuild.yaml``.
    """
    location = pipeline.image.location or project.region
    repo = pipeline.image.repo
    cmd = [
        "gcloud",
        "artifacts",
        "repositories",
        "create",
        repo,
        "--repository-format=docker",
        f"--location={location}",
        f"--project={project.id}",
        f"--labels=solution={SOLUTION}",
        "--description=geapTimes runtime container images",
    ]
    if dry_run:
        logger.info(
            "[dry-run] would ensure Artifact Registry repo %s in %s (%s)",
            repo,
            location,
            " ".join(cmd),
        )
        return
    code, _out, err = (runner or _run_command)(cmd)
    if code == 0:
        logger.info("Created Artifact Registry repo %s in %s", repo, location)
    elif "ALREADY_EXISTS" in err or "already exists" in err.lower():
        logger.info("Artifact Registry repo %s already exists", repo)
    else:
        msg = f"failed to ensure Artifact Registry repo {repo!r}: {err.strip()}"
        raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> int:
    """Parse args, load config, and provision resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without contacting GCP."
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    logger.info("Provisioning for project %s (region %s)", cfg.project.id, cfg.project.region)
    ensure_dataset(cfg.project, dry_run=args.dry_run)
    ensure_bucket(cfg.project, dry_run=args.dry_run)
    ensure_artifact_registry(cfg.project, cfg.pipeline, dry_run=args.dry_run)
    logger.info("Done.")
    return 0
