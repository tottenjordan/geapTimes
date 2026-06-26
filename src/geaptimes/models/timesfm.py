"""TimesFM 2.5 forecaster — an in-process, zero-shot foundation model.

TimesFM needs no training: we load the pretrained checkpoint, ``compile`` it once with a
:class:`timesfm.ForecastConfig`, then call ``forecast(horizon, inputs)`` with one context array
per series. This module pulls each series' recent history from the prepped table and maps the
model's point + quantile output onto the standardized :data:`~geaptimes.models.base.ForecastResult`
shape.

The checkpoint load and BigQuery read are both deferred/injectable so the class imports and
unit-tests without downloading the 200M-parameter model or touching the cloud.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from geaptimes.models.base import (
    PREDICTION_COLUMNS,
    Forecaster,
    ForecastResult,
)
from geaptimes.models.timesfm_core import (
    _quantile_channels,
    compile_timesfm,
    forecast_arrays,
    rows_from_forecast,
)
from geaptimes.naming import table_names
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from geaptimes.schemas import ExperimentConfig, ModelConfig, TimesFMParams

logger = get_logger(__name__)

# Re-exported for backwards-compatible imports; canonical definition in timesfm_core.
__all__ = ["TimesFMForecaster", "_quantile_channels"]


class TimesFMForecaster(Forecaster):
    """Univariate TimesFM 2.5 forecaster (covariates opt-in/deferred)."""

    def __init__(
        self,
        model_cfg: "ModelConfig",
        cfg: "ExperimentConfig",
        *,
        model_loader: "Callable[[str], Any] | None" = None,
        frame_loader: "Callable[[], pd.DataFrame] | None" = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(model_cfg, cfg)
        self.params: TimesFMParams = cast("TimesFMParams", model_cfg.params)
        self._model_loader = model_loader or self._default_model_loader
        self._frame_loader = frame_loader or self._default_frame_loader
        self._cuda_available = cuda_available or _torch_cuda_available
        self._model: Any | None = None
        self.device = self._resolve_device()

    # -- device -----------------------------------------------------------------------------
    def _resolve_device(self) -> str:
        """Pick the torch device from the execution target, falling back to CPU."""
        if self.cfg.execution.target == "local_gpu":
            if self._cuda_available():
                return "cuda"
            logger.warning("local_gpu requested but CUDA is unavailable; falling back to cpu")
        return "cpu"

    # -- model ------------------------------------------------------------------------------
    def _default_model_loader(self, checkpoint: str) -> Any:  # noqa: ANN401 - external model
        """Load + compile the real TimesFM 2.5 model (deferred import; not run in Stage 2)."""
        import timesfm  # noqa: PLC0415 - heavy optional import, kept lazy

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
        return compile_timesfm(
            model,
            max_context=self.params.max_context,
            horizon=self.cfg.forecast.horizon,
            quantiles=self.params.quantiles,
        )

    def _load_model(self) -> Any:  # noqa: ANN401 - external model
        if self._model is None:
            logger.info("loading TimesFM checkpoint %s on %s", self.params.checkpoint, self.device)
            self._model = self._model_loader(self.params.checkpoint)
        return self._model

    # -- data -------------------------------------------------------------------------------
    def _default_frame_loader(self) -> "pd.DataFrame":
        """Read the prepped table (series/date/target/splits) from BigQuery (lazy import)."""
        from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

        data = self.cfg.data
        table = table_names(self.cfg)["prepped"]
        prepped = f"{self.cfg.project.id}.{self.cfg.project.bq_dataset}.{table}"
        # SQL is built from trusted config (column/table names), not user input.
        cols = f"{data.series_column}, {data.time_column}, {data.target_column}, splits"
        sql = f"SELECT {cols} FROM `{prepped}`"  # noqa: S608
        client = bigquery.Client(project=self.cfg.project.id)
        return client.query(sql).to_dataframe()

    # -- interface --------------------------------------------------------------------------
    def fit(self) -> None:
        """No-op: TimesFM is zero-shot."""

    def predict(self) -> ForecastResult:
        """Forecast ``horizon`` steps for every series from its recent context."""
        import pandas as pd  # noqa: PLC0415 - lazy heavy import

        data = self.cfg.data
        horizon = self.cfg.forecast.horizon
        context_len = self.params.context_len

        frame = self._frame_loader()
        train = frame[frame["splits"] != "TEST"]

        series_ids: list[Any] = []
        inputs: list[Any] = []
        last_dates: list[Any] = []
        for series_id, group in train.groupby(data.series_column, sort=True):
            ordered = group.sort_values(data.time_column)
            history = ordered[data.target_column].to_numpy(dtype=float)[-context_len:]
            series_ids.append(series_id)
            inputs.append(history)
            last_dates.append(pd.Timestamp(ordered[data.time_column].iloc[-1]))

        model = self._load_model()
        forecast = forecast_arrays(model, inputs, horizon)
        rows = rows_from_forecast(
            series_ids, last_dates, forecast, horizon=horizon, model_name=self.name
        )

        predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        metadata = {
            "checkpoint": self.params.checkpoint,
            "device": self.device,
            "context_len": context_len,
            "n_series": len(series_ids),
        }
        return ForecastResult(predictions=predictions, metadata=metadata)


def _torch_cuda_available() -> bool:
    """Lazily probe torch for CUDA without importing it at module load."""
    try:
        import torch  # noqa: PLC0415 - lazy heavy import
    except ImportError:  # pragma: no cover - torch is a hard dep, defensive only
        return False
    return bool(torch.cuda.is_available())
