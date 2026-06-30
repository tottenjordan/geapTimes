"""CLI entry point to run the in-process experiment for a geapTimes config.

Thin wrapper around :mod:`geaptimes.experiment.cli`.

Usage:
    uv run python scripts/run_experiment.py --config config/base_config.yaml [--dry-run]
    uv run python scripts/run_experiment.py --config config/comparison_config.yaml --disable-automl
"""

from geaptimes.experiment.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
