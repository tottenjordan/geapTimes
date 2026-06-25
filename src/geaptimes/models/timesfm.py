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
    DATE_COLUMN,
    FORECAST_COLUMN,
    HORIZON_STEP_COLUMN,
    MODEL_COLUMN,
    PREDICTION_COLUMNS,
    QUANTILES,
    SERIES_COLUMN,
    Forecaster,
    ForecastResult,
    quantile_column,
)
from geaptimes.naming import table_names
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from geaptimes.schemas import ExperimentConfig, ModelConfig, TimesFMParams

logger = get_logger(__name__)

# TimesFM's continuous-quantile head emits 10 channels: index 0 is the mean, indices 1..9 are
# the deciles q10..q90. So quantile level ``q`` lives at channel ``round(q * 10)``.
_DECILE_CHANNELS = 10


def _quantile_channels(n_channels: int) -> list[int]:
    """Channel index per :data:`QUANTILES` level for the model's quantile output."""
    if n_channels >= _DECILE_CHANNELS:
        return [round(q * 10) for q in QUANTILES]
    # Fallback: a head that emits exactly our levels positionally.
    return list(range(len(QUANTILES)))


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
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.params.max_context,
                max_horizon=self.cfg.forecast.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=self.params.quantiles,
                fix_quantile_crossing=self.params.quantiles,
            )
        )
        return model

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
        import numpy as np  # noqa: PLC0415 - lazy heavy import
        import pandas as pd  # noqa: PLC0415 - lazy heavy import

        data = self.cfg.data
        horizon = self.cfg.forecast.horizon
        context_len = self.params.context_len

        frame = self._frame_loader()
        train = frame[frame["splits"] != "TEST"]

        series_ids: list[Any] = []
        inputs: list[np.ndarray] = []
        last_dates: list[Any] = []
        for series_id, group in train.groupby(data.series_column, sort=True):
            ordered = group.sort_values(data.time_column)
            history = ordered[data.target_column].to_numpy(dtype=float)[-context_len:]
            series_ids.append(series_id)
            inputs.append(history)
            last_dates.append(pd.Timestamp(ordered[data.time_column].iloc[-1]))

        model = self._load_model()
        point, quantiles = model.forecast(horizon, inputs=inputs)
        point = np.asarray(point)
        quantiles = np.asarray(quantiles)
        channels = _quantile_channels(quantiles.shape[-1])

        rows: list[dict[str, Any]] = []
        for i, series_id in enumerate(series_ids):
            for step in range(horizon):
                row: dict[str, Any] = {
                    SERIES_COLUMN: series_id,
                    DATE_COLUMN: (last_dates[i] + pd.Timedelta(days=step + 1)).date(),
                    HORIZON_STEP_COLUMN: step + 1,
                    FORECAST_COLUMN: float(point[i, step]),
                    MODEL_COLUMN: self.name,
                }
                for level, channel in zip(QUANTILES, channels, strict=True):
                    row[quantile_column(level)] = float(quantiles[i, step, channel])
                rows.append(row)

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
