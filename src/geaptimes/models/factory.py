"""Factory that builds the right :class:`~geaptimes.models.base.Forecaster` from config.

Dispatch mirrors the schema's discriminated union on ``params.type`` so a new backend is wired in
one place. Forecaster modules are imported lazily inside :meth:`ForecastFactory.create` so that
importing the factory never pulls in heavy/optional backend dependencies until a model is built.
"""

from typing import TYPE_CHECKING

from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geaptimes.models.base import Forecaster
    from geaptimes.schemas import ExperimentConfig, ModelConfig

logger = get_logger(__name__)

# Supported backend discriminators (mirrors geaptimes.schemas.ModelParams).
_BACKENDS = ("timesfm", "bqml", "automl")


class ForecastFactory:
    """Construct forecasters from validated config."""

    @staticmethod
    def create(model_cfg: "ModelConfig", cfg: "ExperimentConfig") -> "Forecaster":
        """Build a single forecaster for ``model_cfg`` (dispatch on ``params.type``)."""
        backend = model_cfg.params.type
        if backend == "timesfm":
            from geaptimes.models.timesfm import TimesFMForecaster  # noqa: PLC0415 - lazy backend

            return TimesFMForecaster(model_cfg, cfg)
        if backend == "bqml":
            from geaptimes.models.bqml import BQMLForecaster  # noqa: PLC0415 - lazy backend

            return BQMLForecaster(model_cfg, cfg)
        if backend == "automl":
            from geaptimes.models.automl import AutoMLForecaster  # noqa: PLC0415 - lazy backend

            return AutoMLForecaster(model_cfg, cfg)
        msg = f"Unknown forecaster backend {backend!r}; expected one of {_BACKENDS}"
        raise ValueError(msg)

    @staticmethod
    def from_config(cfg: "ExperimentConfig") -> list["Forecaster"]:
        """Build forecasters for every **enabled** model in the experiment config."""
        forecasters: list[Forecaster] = []
        for model_cfg in cfg.models:
            if not model_cfg.enabled:
                logger.info("skipping disabled model %s", model_cfg.name)
                continue
            forecasters.append(ForecastFactory.create(model_cfg, cfg))
        return forecasters
