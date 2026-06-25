"""Tests for the GCP provisioning script (dry-run only; no network)."""

import logging
from pathlib import Path

from geaptimes import gcp

CONFIG_PATH = Path(__file__).parents[1] / "config" / "base_config.yaml"


def test_dry_run_logs_plan_without_network(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GCP_PROJECT", "unit-test-proj")
    with caplog.at_level(logging.INFO, logger="geaptimes.setup_gcp"):
        rc = gcp.main(["--config", str(CONFIG_PATH), "--dry-run"])

    assert rc == 0
    messages = " ".join(caplog.messages)
    assert "[dry-run] would ensure BigQuery dataset unit-test-proj.geaptimes" in messages
    assert "[dry-run] would ensure GCS bucket gs://geaptimes-unit-test-proj" in messages
