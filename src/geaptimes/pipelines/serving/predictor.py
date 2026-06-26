"""TimesFM serving predictor — the model logic behind the custom serving container.

:class:`TimesFMPredictor` loads + compiles the checkpoint once and maps prediction requests
(one context array per series) to the standardized forecast arrays via
:mod:`geaptimes.models.timesfm_core`. The model loader is injected, so the predictor unit-tests
offline with a fake model (no checkpoint download, no torch).
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from geaptimes.models.timesfm_core import compile_timesfm, forecast_arrays, series_quantiles
from geaptimes.utils.logger import get_logger

logger = get_logger(__name__)

ModelLoader = Callable[["PredictorConfig"], Any]


class Predictor(Protocol):
    """Structural type for anything the serving app can call (real predictor or a test fake)."""

    def predict(self, instances: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PredictorConfig:
    """Serving-time TimesFM settings (mirrors the relevant TimesFMParams + forecast horizon)."""

    checkpoint: str = "google/timesfm-2.5-200m-pytorch"
    max_context: int = 1024
    context_len: int = 512
    horizon: int = 14
    quantiles: bool = True
    model_name: str = "timesfm"

    @classmethod
    def from_env(cls) -> "PredictorConfig":
        """Build the config from the container's environment (deploy-time overrides)."""
        return cls(
            checkpoint=os.environ.get("TIMESFM_CHECKPOINT", cls.checkpoint),
            max_context=int(os.environ.get("TIMESFM_MAX_CONTEXT", cls.max_context)),
            context_len=int(os.environ.get("TIMESFM_CONTEXT_LEN", cls.context_len)),
            horizon=int(os.environ.get("TIMESFM_HORIZON", cls.horizon)),
            quantiles=os.environ.get("TIMESFM_QUANTILES", "true").lower() != "false",
            model_name=os.environ.get("TIMESFM_MODEL_NAME", cls.model_name),
        )


def _default_model_loader(config: PredictorConfig) -> Any:  # noqa: ANN401 - external model
    """Load + compile the real TimesFM 2.5 model (deferred import; heavy)."""
    import timesfm  # noqa: PLC0415 - heavy optional import, kept lazy

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config.checkpoint)
    return compile_timesfm(
        model,
        max_context=config.max_context,
        horizon=config.horizon,
        quantiles=config.quantiles,
    )


class TimesFMPredictor:
    """Serve TimesFM forecasts: ``predict(instances) -> responses``."""

    def __init__(
        self,
        config: PredictorConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.config = config or PredictorConfig()
        self._model_loader = model_loader or _default_model_loader
        self._model: Any | None = None

    @classmethod
    def from_env(cls, *, model_loader: ModelLoader | None = None) -> "TimesFMPredictor":
        """Construct from environment-driven config (used by the serving container)."""
        return cls(PredictorConfig.from_env(), model_loader=model_loader)

    def _load(self) -> Any:  # noqa: ANN401 - external model
        if self._model is None:
            logger.info("loading TimesFM checkpoint %s for serving", self.config.checkpoint)
            self._model = self._model_loader(self.config)
        return self._model

    def predict(self, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Forecast each instance's series from its context array.

        Each instance is ``{"series_id": ..., "context": [float, ...]}``; each response is
        ``{"series_id", "horizon", "forecast": [...], "q10": [...], ..., "q90": [...]}``.
        """
        import numpy as np  # noqa: PLC0415 - lazy heavy import

        if not instances:
            return []

        series_ids = [inst.get("series_id") for inst in instances]
        inputs = [
            np.asarray(inst["context"], dtype=float)[-self.config.context_len :]
            for inst in instances
        ]
        model = self._load()
        point, quantiles = forecast_arrays(model, inputs, self.config.horizon)

        responses: list[dict[str, Any]] = []
        for i, series_id in enumerate(series_ids):
            arrays = series_quantiles(point[i], quantiles[i], horizon=self.config.horizon)
            responses.append({"series_id": series_id, "horizon": self.config.horizon, **arrays})
        return responses
