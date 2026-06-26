"""Tests for the GCP provisioning script (dry-run + injected runner; no network)."""

import logging
from pathlib import Path

import pytest

from geaptimes import gcp
from geaptimes.schemas import PipelineConfig, ProjectConfig

CONFIG_PATH = Path(__file__).parents[1] / "config" / "base_config.yaml"


def test_dry_run_logs_plan_without_network(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GCP_PROJECT", "unit-test-proj")
    with caplog.at_level(logging.INFO, logger="geaptimes.setup_gcp"):
        rc = gcp.main(["--config", str(CONFIG_PATH), "--dry-run"])

    assert rc == 0
    messages = " ".join(caplog.messages)
    assert "[dry-run] would ensure BigQuery dataset unit-test-proj.geaptimes" in messages
    assert "[dry-run] would ensure GCS bucket gs://geaptimes-unit-test-proj" in messages
    assert "[dry-run] would ensure Artifact Registry repo geaptimes in us-central1" in messages


_PROJECT = ProjectConfig(id="p", gcs_bucket="b", region="us-central1")


def test_ensure_artifact_registry_creates_with_labels() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> tuple[int, str, str]:
        calls.append(cmd)
        return 0, "Created repo", ""

    gcp.ensure_artifact_registry(_PROJECT, PipelineConfig(), dry_run=False, runner=runner)
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:4] == ["gcloud", "artifacts", "repositories", "create"]
    assert "geaptimes" in cmd
    assert "--repository-format=docker" in cmd
    assert "--location=us-central1" in cmd
    assert "--labels=solution=geaptimes" in cmd


def test_ensure_artifact_registry_idempotent_on_already_exists() -> None:
    def runner(_cmd: list[str]) -> tuple[int, str, str]:
        return 1, "", "ERROR: ALREADY_EXISTS: the repository already exists"

    # Must not raise when the repo is already there.
    gcp.ensure_artifact_registry(_PROJECT, PipelineConfig(), dry_run=False, runner=runner)


def test_ensure_artifact_registry_raises_on_real_error() -> None:
    def runner(_cmd: list[str]) -> tuple[int, str, str]:
        return 1, "", "ERROR: PERMISSION_DENIED"

    with pytest.raises(RuntimeError, match="failed to ensure Artifact Registry repo"):
        gcp.ensure_artifact_registry(_PROJECT, PipelineConfig(), dry_run=False, runner=runner)
