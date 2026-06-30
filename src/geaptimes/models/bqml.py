"""BigQuery ML ARIMA forecaster (``ARIMA_PLUS`` / ``ARIMA_PLUS_XREG``).

Training and inference both run server-side in BigQuery: ``fit()`` issues a ``CREATE MODEL`` and
``predict()`` issues ``ML.FORECAST``. For ``ARIMA_PLUS_XREG`` the future regressor values are
supplied as ``ML.FORECAST``'s third table argument (the ``infer`` table built in
:mod:`geaptimes.data.queries`).

BQML emits a point ``forecast_value`` plus a Gaussian ``standard_error``; we derive the
standardized decile columns (:data:`~geaptimes.models.base.QUANTILES`) as
``forecast_value + z(level) * standard_error`` using fixed z-scores, so no extra dependency is
needed. The BigQuery client is injected (lazy default) so the class unit-tests without executing
any query.
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
from geaptimes.naming import config_slug
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    import pandas as pd

    from geaptimes.schemas import BQMLParams, DataConfig, ExperimentConfig, ModelConfig

logger = get_logger(__name__)

# Standard-normal quantiles for the fixed QUANTILES levels (z such that Phi(z) = level).
_Z_SCORES: dict[float, float] = {
    0.1: -1.2815515594,
    0.3: -0.5244005127,
    0.5: 0.0,
    0.7: 0.5244005127,
    0.9: 1.2815515594,
}

_DEFAULT_CONFIDENCE_LEVEL = 0.9

# ML.FORECAST output column names.
_FC_TIMESTAMP = "forecast_timestamp"
_FC_VALUE = "forecast_value"
_FC_STD_ERR = "standard_error"


def _xreg_columns(data: "DataConfig", params: "BQMLParams") -> list[str]:
    """Future regressors for XREG: the available-at-forecast covariates (empty for plain ARIMA)."""
    if params.model_type == "ARIMA_PLUS_XREG" and params.use_covariates:
        return list(data.covariates.available_at_forecast_columns)
    return []


def build_create_model_sql(
    cfg: "ExperimentConfig",
    params: "BQMLParams",
    *,
    model_id: str,
    train_table: str,
) -> str:
    """Build the ``CREATE OR REPLACE MODEL`` statement for a BQML ARIMA model."""
    data = cfg.data
    regressors = _xreg_columns(data, params)
    select_cols = [data.series_column, data.time_column, data.target_column, *regressors]
    options = [
        f"model_type = '{params.model_type}'",
        f"time_series_timestamp_col = '{data.time_column}'",
        f"time_series_data_col = '{data.target_column}'",
        f"time_series_id_col = '{data.series_column}'",
        "data_frequency = 'DAILY'",
    ]
    if data.holiday_region:
        options.append(f"holiday_region = '{data.holiday_region}'")
    # NB: BQML CREATE MODEL does not accept resource labels in OPTIONS (the `labels` key collides
    # with the training-label-column semantics, typed ARRAY<STRING>). The solution=geaptimes label
    # is applied to the model resource after creation via the BigQuery API (see fit()).
    options_sql = ",\n    ".join(options)
    cols_sql = ", ".join(select_cols)
    return f"""CREATE OR REPLACE MODEL `{model_id}`
OPTIONS(
    {options_sql}
) AS
SELECT {cols_sql}
FROM `{train_table}`
"""  # noqa: S608 - built from trusted config, not user input


def build_forecast_sql(
    cfg: "ExperimentConfig",
    params: "BQMLParams",
    *,
    model_id: str,
    infer_table: str,
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
) -> str:
    """Build the ``ML.FORECAST`` statement (XREG joins the infer table for future regressors)."""
    data = cfg.data
    horizon = cfg.forecast.horizon
    struct = f"STRUCT({horizon} AS horizon, {confidence_level} AS confidence_level)"
    regressors = _xreg_columns(data, params)
    if regressors:
        future_cols = ", ".join([data.series_column, data.time_column, *regressors])
        future = f",\n    (SELECT {future_cols} FROM `{infer_table}`)"  # noqa: S608 - trusted config
    else:
        future = ""
    return f"""SELECT * FROM ML.FORECAST(
    MODEL `{model_id}`,
    {struct}{future}
)
"""  # noqa: S608 - built from trusted config, not user input


class BQMLForecaster(Forecaster):
    """BigQuery ML ARIMA_PLUS / ARIMA_PLUS_XREG forecaster (runs server-side)."""

    def __init__(
        self,
        model_cfg: "ModelConfig",
        cfg: "ExperimentConfig",
        *,
        query_runner: "Callable[[str], pd.DataFrame] | None" = None,
        model_labeler: "Callable[[str, dict[str, str]], None] | None" = None,
    ) -> None:
        super().__init__(model_cfg, cfg)
        self.params: BQMLParams = cast("BQMLParams", model_cfg.params)
        self._query_runner = query_runner or self._default_query_runner
        self._model_labeler = model_labeler or self._default_model_labeler
        slug = config_slug(cfg.data, cfg.forecast)
        self.model_id = f"{cfg.project.id}.{cfg.project.bq_dataset}.{model_cfg.name}__{slug}"

    def _table(self, role: str) -> str:
        from geaptimes.naming import table_names  # noqa: PLC0415 - avoid import cycle at top

        name = table_names(self.cfg)[role]
        return f"{self.cfg.project.id}.{self.cfg.project.bq_dataset}.{name}"

    def _default_query_runner(self, sql: str) -> "pd.DataFrame":
        """Execute SQL in BigQuery and return the result frame (lazy import; not run in Stage 2)."""
        from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

        client = bigquery.Client(project=self.cfg.project.id)
        return client.query(sql).to_dataframe()

    def _default_model_labeler(self, model_id: str, labels: dict[str, str]) -> None:
        """Apply resource labels to the trained model via the BigQuery API (not run in Stage 2)."""
        from google.cloud import bigquery  # noqa: PLC0415 - lazy cloud import

        client = bigquery.Client(project=self.cfg.project.id)
        model = client.get_model(model_id)
        model.labels = labels
        client.update_model(model, ["labels"])

    @property
    def model_reference(self) -> str:
        """The fully-qualified BQML model id; config-derived, so stable without a live ``fit()``."""
        return self.model_id

    # attach_model is the inherited no-op: predict() already references the model by id (the same
    # config-derived self.model_id), so no live handle needs to be restored between split steps.

    def fit(self) -> None:
        """Create (train) the ARIMA model in BigQuery and apply resource labels."""
        sql = build_create_model_sql(
            self.cfg,
            self.params,
            model_id=self.model_id,
            train_table=self._table("train"),
        )
        logger.info("creating BQML model %s", self.model_id)
        self._query_runner(sql)
        # BQML CREATE MODEL can't carry resource labels in OPTIONS; label the model resource here.
        logger.info("labeling BQML model %s", self.model_id)
        self._model_labeler(self.model_id, dict(RESOURCE_LABELS))

    def predict(self) -> ForecastResult:
        """Run ``ML.FORECAST`` and map the result onto the standardized prediction frame."""
        import pandas as pd  # noqa: PLC0415 - lazy heavy import

        sql = build_forecast_sql(
            self.cfg,
            self.params,
            model_id=self.model_id,
            infer_table=self._table("infer"),
        )
        raw = self._query_runner(sql)
        predictions = self._to_predictions(raw, pd)
        metadata = {"bq_model_id": self.model_id, "model_type": self.params.model_type}
        return ForecastResult(predictions=predictions, metadata=metadata)

    def _to_predictions(self, raw: "pd.DataFrame", pd: Any) -> "pd.DataFrame":  # noqa: ANN401
        """Map ML.FORECAST output → standardized columns (deciles from Gaussian std error)."""
        series_col = self.cfg.data.series_column
        out = pd.DataFrame()
        out[SERIES_COLUMN] = raw[series_col]
        out[DATE_COLUMN] = pd.to_datetime(raw[_FC_TIMESTAMP]).dt.date
        out[FORECAST_COLUMN] = raw[_FC_VALUE].astype(float)
        for level in QUANTILES:
            out[quantile_column(level)] = raw[_FC_VALUE].astype(float) + _Z_SCORES[level] * raw[
                _FC_STD_ERR
            ].astype(float)
        out[MODEL_COLUMN] = self.name
        # horizon_step = per-series rank by forecast date (1-based).
        out[HORIZON_STEP_COLUMN] = (
            out.groupby(SERIES_COLUMN)[DATE_COLUMN].rank(method="first").astype(int)
        )
        return out[PREDICTION_COLUMNS]
