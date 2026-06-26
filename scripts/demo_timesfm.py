"""Local TimesFM 2.5 forecast demo.

Builds the TimesFM forecaster from a config via :class:`~geaptimes.models.factory.ForecastFactory`,
runs a zero-shot forecast over the horizon, and prints the result. Reads the prepped table from
BigQuery and downloads the checkpoint on first use, so it needs auth + network.

Usage:
    uv run python scripts/demo_timesfm.py --config config/base_config.yaml [--model NAME] [--rows N]
"""

import argparse

from geaptimes.models.factory import ForecastFactory
from geaptimes.schemas import ExperimentConfig
from geaptimes.utils.logger import get_logger

logger = get_logger("geaptimes.demo")


def _pick_model(cfg: ExperimentConfig, name: str | None) -> str:
    """Return the requested model name, or the first enabled TimesFM model."""
    if name is not None:
        return name
    for model in cfg.models:
        if model.enabled and model.params.type == "timesfm":
            return model.name
    msg = "no enabled TimesFM model found in config; pass --model"
    raise SystemExit(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local TimesFM 2.5 forecast demo")
    parser.add_argument("--config", default="config/base_config.yaml", help="experiment YAML")
    parser.add_argument("--model", default=None, help="model name (default: first enabled timesfm)")
    parser.add_argument("--rows", type=int, default=10, help="prediction rows to print")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    model_name = _pick_model(cfg, args.model)
    model_cfg = next(m for m in cfg.models if m.name == model_name)

    forecaster = ForecastFactory.create(model_cfg, cfg)
    logger.info("running %s (horizon=%d)", model_name, cfg.forecast.horizon)
    forecaster.fit()  # no-op for TimesFM (zero-shot)
    result = forecaster.predict()

    logger.info("metadata: %s", result.metadata)
    logger.info(
        "predictions (%d rows):\n%s", len(result.predictions), result.predictions.head(args.rows)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
