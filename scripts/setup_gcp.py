"""CLI entry point to provision GCP resources for a geapTimes config.

Thin wrapper around :mod:`geaptimes.gcp`.

Usage:
    uv run python scripts/setup_gcp.py --config config/base_config.yaml [--dry-run]
"""

from geaptimes.gcp import main

if __name__ == "__main__":
    raise SystemExit(main())
