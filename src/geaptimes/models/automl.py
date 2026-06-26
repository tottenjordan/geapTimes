"""Vertex AI AutoML forecasting wrapper.

The substantive job of this wrapper is translating the typed config into the
``AutoMLForecastingTrainingJob`` SDK surface: the covariate **column roles** (attribute /
available-at-forecast / unavailable-at-forecast, with the target itself marked unavailable as Vertex
requires), the explicit **column transformations** (``auto`` for every used column *including the
target* -- the SDK's default omits the target, which Vertex Forecasting then rejects), the forecast
horizon and context window, the daily granularity, Vertex's default chronological split over the
TEST-free train table, the standardized :data:`~geaptimes.models.base.QUANTILES`, holiday regions,
and the ``solution=geaptimes`` labels.

Training and batch prediction run on Vertex (long-running, billable), so ``aiplatform`` is an
injected seam — the kwargs builders are pure and unit-tested offline, and Stage 2 never launches a
job. The first live AutoML run happens in Stage 4, which also confirms the nested batch-prediction
output struct flattened by :meth:`AutoMLForecaster._flatten_automl_output`.
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
# Column emitted by Vertex batch prediction holding the point forecast (flattened by the reader).
_PREDICTED_VALUE_COLUMN = "value"
# Vertex forecasting batch output nests results in a STRUCT column named predicted_<target> with
# fields: value (point), quantile_values (the trained levels), quantile_predictions (parallel).
_QUANTILE_VALUES_FIELD = "quantile_values"
_QUANTILE_PREDICTIONS_FIELD = "quantile_predictions"
_QUANTILE_MATCH_TOLERANCE = 1e-6


def _match_quantile_index(levels: "list[float]", level: float) -> int:
    """Index of ``level`` within the model's emitted quantile levels (float-tolerant)."""
    for i, candidate in enumerate(levels):
        if abs(float(candidate) - level) < _QUANTILE_MATCH_TOLERANCE:
            return i
    msg = f"quantile level {level} not found in AutoML output levels {levels}"
    raise ValueError(msg)


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

    def _column_specs(self) -> dict[str, str]:
        """Column-type transformations for every column the model uses.

        The SDK's default (passing neither ``column_specs`` nor ``column_transformations``) auto-
        generates an ``auto`` transformation for every dataset column *except the target* -- but
        Vertex Forecasting then rejects the job because the target carries no transformation
        ("The target column ... does not have a numeric or auto transformation."). So we build the
        specs ourselves, mirroring that default ``auto`` behaviour while *including* the target.
        Columns are the union of the target and every role column (time, series id, and the three
        covariate groups); duplicates collapse in the dict.
        """
        data = self.cfg.data
        covariates = data.covariates
        columns = [
            data.target_column,
            data.time_column,
            data.series_column,
            *covariates.time_series_attribute_columns,
            *covariates.available_at_forecast_columns,
            *covariates.unavailable_at_forecast_columns,
        ]
        return dict.fromkeys(columns, "auto")

    def training_job_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs for ``AutoMLForecastingTrainingJob``."""
        return {
            "display_name": self.display_name,
            "optimization_objective": self.params.optimization_objective,
            "column_specs": self._column_specs(),
            "labels": dict(RESOURCE_LABELS),
        }

    def run_kwargs(self) -> dict[str, Any]:
        """``.run()`` kwargs: column roles, horizon/context, granularity, split, quantiles."""
        data = self.cfg.data
        covariates = data.covariates
        # Vertex AutoML requires the target column itself to be declared unavailable-at-forecast
        # (its future value is unknown at prediction time); add it if not already listed.
        unavailable = list(covariates.unavailable_at_forecast_columns)
        if data.target_column not in unavailable:
            unavailable.append(data.target_column)
        return {
            "target_column": data.target_column,
            "time_column": data.time_column,
            "time_series_identifier_column": data.series_column,
            "time_series_attribute_columns": list(covariates.time_series_attribute_columns),
            "available_at_forecast_columns": list(covariates.available_at_forecast_columns),
            "unavailable_at_forecast_columns": unavailable,
            "forecast_horizon": self.cfg.forecast.horizon,
            "context_window": self.params.context_window,
            "budget_milli_node_hours": self.params.budget_milli_node_hours,
            "data_granularity_unit": _GRANULARITY_UNIT,
            "data_granularity_count": _GRANULARITY_COUNT,
            # No predefined split: the training dataset is the TEST-free train table, so Vertex's
            # default chronological auto-split is used for AutoML's internal validation. (A
            # predefined `splits` column can't be used here: Vertex requires the values
            # TRAIN/VALIDATION/TEST, but the train table has no TEST rows by design.) The backtest
            # stays honest -- AutoML never sees the TEST window, which we score via batch predict.
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
        flat = self._flatten_automl_output(raw, pd)
        predictions = self._to_predictions(flat, pd)
        metadata = {
            "job_display_name": self.display_name,
            "objective": self.params.optimization_objective,
        }
        return ForecastResult(predictions=predictions, metadata=metadata)

    def _read_predictions(self) -> "pd.DataFrame":
        """Submit batch prediction and return the raw predictions frame."""
        if self._prediction_reader is not None:
            return self._prediction_reader()
        if self.model is None:  # pragma: no cover - guard for unfitted live use
            msg = "AutoMLForecaster.predict() requires fit() (or an injected prediction_reader)"
            raise RuntimeError(msg)
        job = self.model.batch_predict(**self.batch_predict_kwargs())  # pragma: no cover - live
        return self._read_output_table(job)  # pragma: no cover - live only

    def _read_output_table(self, job: Any) -> "pd.DataFrame":  # noqa: ANN401 - external SDK job
        """Read the BQ table Vertex wrote the batch predictions to (resolved from the job)."""
        from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

        dataset = job.output_info.bigquery_output_dataset.removeprefix("bq://")
        table = f"{dataset}.predictions"
        # Table name resolved from the Vertex job output, not user input.
        sql = f"SELECT * FROM `{table}`"  # noqa: S608
        client = bigquery.Client(project=self.cfg.project.id)
        return client.query(sql).to_dataframe()

    def _flatten_automl_output(self, raw: "pd.DataFrame", pd: Any) -> "pd.DataFrame":  # noqa: ANN401
        """Flatten Vertex's nested ``predicted_<target>`` struct to flat value + quantile columns.

        Already-flat frames (e.g. an injected ``prediction_reader``) pass through unchanged. The
        nested struct exposes ``value`` (point), ``quantile_values`` (the trained levels), and the
        parallel ``quantile_predictions``; we select the standardized :data:`QUANTILES` by matching
        levels. The exact field names are confirmed on the first live run (Stage 4).
        """
        data = self.cfg.data
        struct_col = f"predicted_{data.target_column}"
        if struct_col not in raw.columns:
            return raw

        records: list[dict[str, Any]] = []
        for _, row in raw.iterrows():
            struct = row[struct_col]
            levels = [float(level) for level in struct[_QUANTILE_VALUES_FIELD]]
            preds = list(struct[_QUANTILE_PREDICTIONS_FIELD])
            record: dict[str, Any] = {
                data.series_column: row[data.series_column],
                data.time_column: row[data.time_column],
                _PREDICTED_VALUE_COLUMN: float(struct[_PREDICTED_VALUE_COLUMN]),
            }
            for level in QUANTILES:
                record[quantile_column(level)] = float(preds[_match_quantile_index(levels, level)])
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _to_predictions(self, raw: "pd.DataFrame", pd: Any) -> "pd.DataFrame":  # noqa: ANN401
        """Map the flattened batch-prediction frame onto :data:`PREDICTION_COLUMNS`.

        Expects series/time keys, a point ``value`` column, and one column per standardized
        quantile (``q10``..``q90``) — i.e. the output of :meth:`_flatten_automl_output`.
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
