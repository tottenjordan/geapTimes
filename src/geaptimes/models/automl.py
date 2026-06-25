"""Vertex AI AutoML forecasting wrapper.

The substantive job of this wrapper is translating the typed config into the
``AutoMLForecastingTrainingJob`` SDK surface: the covariate **column roles** (attribute /
available-at-forecast / unavailable-at-forecast), the forecast horizon and context window, the
daily granularity, the ``splits`` predefined-split column, the standardized
:data:`~geaptimes.models.base.QUANTILES`, holiday regions, and the ``solution=geaptimes`` labels.

Training and batch prediction run on Vertex (long-running, billable), so ``aiplatform`` is an
injected seam — the kwargs builders are pure and unit-tested offline, and Stage 2 never launches a
job. The first live AutoML run happens in Stage 3/4, which also confirms the batch-prediction
output schema consumed by :meth:`AutoMLForecaster._to_predictions`.
"""

from typing import TYPE_CHECKING, Any, cast

from geaptimes.constants import RESOURCE_LABELS
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
from geaptimes.naming import config_slug, table_names
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    import pandas as pd

    from geaptimes.schemas import AutoMLParams, ExperimentConfig, ModelConfig

logger = get_logger(__name__)

_GRANULARITY_UNIT = "day"
_GRANULARITY_COUNT = 1
_SPLIT_COLUMN = "splits"
# Column emitted by Vertex batch prediction holding the point forecast (flattened by the reader).
_PREDICTED_VALUE_COLUMN = "value"


class AutoMLForecaster(Forecaster):
    """Vertex AI AutoML forecasting model (managed training + batch prediction)."""

    def __init__(
        self,
        model_cfg: "ModelConfig",
        cfg: "ExperimentConfig",
        *,
        aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
        prediction_reader: "Callable[[], pd.DataFrame] | None" = None,
    ) -> None:
        super().__init__(model_cfg, cfg)
        self.params: AutoMLParams = cast("AutoMLParams", model_cfg.params)
        self._aiplatform = aiplatform
        self._prediction_reader = prediction_reader
        self.display_name = f"{model_cfg.name}__{config_slug(cfg.data, cfg.forecast)}"
        self.model: Any = None

    # -- config translation (pure, fully tested) --------------------------------------------
    def _bq_uri(self, role: str) -> str:
        name = table_names(self.cfg)[role]
        return f"bq://{self.cfg.project.id}.{self.cfg.project.bq_dataset}.{name}"

    def _holiday_regions(self) -> list[str] | None:
        region = self.cfg.data.holiday_region
        return [region] if region else None

    def training_job_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs for ``AutoMLForecastingTrainingJob``."""
        return {
            "display_name": self.display_name,
            "optimization_objective": self.params.optimization_objective,
            "labels": dict(RESOURCE_LABELS),
        }

    def run_kwargs(self) -> dict[str, Any]:
        """``.run()`` kwargs: column roles, horizon/context, granularity, split, quantiles."""
        data = self.cfg.data
        covariates = data.covariates
        return {
            "target_column": data.target_column,
            "time_column": data.time_column,
            "time_series_identifier_column": data.series_column,
            "time_series_attribute_columns": list(covariates.time_series_attribute_columns),
            "available_at_forecast_columns": list(covariates.available_at_forecast_columns),
            "unavailable_at_forecast_columns": list(covariates.unavailable_at_forecast_columns),
            "forecast_horizon": self.cfg.forecast.horizon,
            "context_window": self.params.context_window,
            "data_granularity_unit": _GRANULARITY_UNIT,
            "data_granularity_count": _GRANULARITY_COUNT,
            "predefined_split_column_name": _SPLIT_COLUMN,
            "quantiles": list(QUANTILES),
            "holiday_regions": self._holiday_regions(),
            "model_display_name": self.display_name,
            "model_labels": dict(RESOURCE_LABELS),
        }

    def batch_predict_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``model.batch_predict`` against the infer table (BQ → BQ)."""
        dataset = f"{self.cfg.project.id}.{self.cfg.project.bq_dataset}"
        return {
            "job_display_name": f"{self.display_name}__predict",
            "bigquery_source": self._bq_uri("infer"),
            "bigquery_destination_prefix": f"bq://{dataset}",
            "instances_format": "bigquery",
            "predictions_format": "bigquery",
            "labels": dict(RESOURCE_LABELS),
        }

    # -- execution seam (injected; not run in Stage 2) --------------------------------------
    def _resolve_aiplatform(self) -> Any:  # noqa: ANN401 - external SDK module
        if self._aiplatform is None:
            from google.cloud import aiplatform  # noqa: PLC0415 - lazy cloud import

            self._aiplatform = aiplatform
        return self._aiplatform

    def fit(self) -> None:
        """Create a TimeSeriesDataset from the train table and launch managed training."""
        aip = self._resolve_aiplatform()
        aip.init(project=self.cfg.project.id, location=self.cfg.project.region)
        dataset = aip.TimeSeriesDataset.create(
            display_name=self.display_name,
            bq_source=self._bq_uri("train"),
            labels=dict(RESOURCE_LABELS),
        )
        job = aip.AutoMLForecastingTrainingJob(**self.training_job_kwargs())
        logger.info("launching AutoML training job %s", self.display_name)
        self.model = job.run(dataset=dataset, **self.run_kwargs())

    def predict(self) -> ForecastResult:
        """Batch-predict over the infer table and map results onto the standardized frame."""
        import pandas as pd  # noqa: PLC0415 - lazy heavy import

        raw = self._read_predictions()
        predictions = self._to_predictions(raw, pd)
        metadata = {
            "job_display_name": self.display_name,
            "objective": self.params.optimization_objective,
        }
        return ForecastResult(predictions=predictions, metadata=metadata)

    def _read_predictions(self) -> "pd.DataFrame":
        """Submit batch prediction and return the (flattened) predictions frame."""
        if self._prediction_reader is not None:
            return self._prediction_reader()
        if self.model is None:  # pragma: no cover - guard for unfitted live use
            msg = "AutoMLForecaster.predict() requires fit() (or an injected prediction_reader)"
            raise RuntimeError(msg)
        self.model.batch_predict(**self.batch_predict_kwargs())  # pragma: no cover - live only
        msg = "Reading the AutoML batch-prediction output table is wired in Stage 4"
        raise NotImplementedError(msg)  # pragma: no cover - live only

    def _to_predictions(self, raw: "pd.DataFrame", pd: Any) -> "pd.DataFrame":  # noqa: ANN401
        """Map the flattened batch-prediction frame onto :data:`PREDICTION_COLUMNS`.

        Expects the reader to expose the series/time keys, a point ``value`` column, and one
        column per standardized quantile (``q10``..``q90``). The exact Vertex output schema is
        confirmed on the first live run (Stage 4).
        """
        data = self.cfg.data
        out = pd.DataFrame()
        out[SERIES_COLUMN] = raw[data.series_column]
        out[DATE_COLUMN] = pd.to_datetime(raw[data.time_column]).dt.date
        out[FORECAST_COLUMN] = raw[_PREDICTED_VALUE_COLUMN].astype(float)
        for level in QUANTILES:
            out[quantile_column(level)] = raw[quantile_column(level)].astype(float)
        out[MODEL_COLUMN] = self.name
        out[HORIZON_STEP_COLUMN] = (
            out.groupby(SERIES_COLUMN)[DATE_COLUMN].rank(method="first").astype(int)
        )
        return out[PREDICTION_COLUMNS]
