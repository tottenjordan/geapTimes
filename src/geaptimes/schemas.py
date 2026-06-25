"""Typed configuration schema for geapTimes experiments.

Experiments are defined in YAML and validated into these Pydantic v2 models. Application code
consumes the validated :class:`ExperimentConfig` (and its members) — never raw YAML.

Per-model parameters use a discriminated union on ``params.type`` so each model carries only its
own, type-checked fields.
"""

import os
import re
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


class _Base(BaseModel):
    """Base model: forbid unknown keys so config typos fail loudly."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ProjectConfig(_Base):
    """GCP project / storage targets."""

    id: str
    region: str = "us-central1"
    gcs_bucket: str
    bq_dataset: str = "geaptimes"


class WeatherConfig(_Base):
    """NOAA GSOD station used for weather covariates (default: NYC LaGuardia)."""

    station_usaf: str = "725030"
    station_wban: str = "14732"


class StationFilter(_Base):
    """How many series (stations) to include, by total trip volume."""

    top_n: int = 25

    @field_validator("top_n")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            msg = "station_filter.top_n must be > 0"
            raise ValueError(msg)
        return v


class DateRange(_Base):
    """Optional ISO date bounds; ``None`` means use the dataset's own min/max."""

    start: str | None = None
    end: str | None = None


class SplitConfig(_Base):
    """Train/validate/test split lengths (statmike-style, off the global max date)."""

    granularity: Literal["DAY"] = "DAY"
    test_length: int = 14
    validate_length: int = 14


class CovariateRoles(_Base):
    """Covariate columns grouped by Vertex AI / GEAP AutoML feature-type role."""

    time_series_attribute_columns: list[str] = Field(default_factory=list)
    available_at_forecast_columns: list[str] = Field(default_factory=list)
    unavailable_at_forecast_columns: list[str] = Field(default_factory=list)


class DataConfig(_Base):
    """Data layer: source tables, target/series/time columns, filters, splits, covariates."""

    trips_table: str = "bigquery-public-data.new_york_citibike.citibike_trips"
    stations_table: str = "bigquery-public-data.new_york.citibike_stations"
    weather_dataset: str = "bigquery-public-data.noaa_gsod"
    series_column: str = "start_station_name"
    time_column: str = "date"
    target_column: str = "num_trips"
    station_filter: StationFilter = Field(default_factory=StationFilter)
    date_range: DateRange = Field(default_factory=DateRange)
    gap_fill: bool = True
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    covariates: CovariateRoles = Field(default_factory=CovariateRoles)
    holiday_region: str | None = "US"


class ForecastSettings(_Base):
    """Canonical forecast settings, mapped to each backend's representation downstream."""

    frequency: Literal["D"] = "D"
    horizon: int = 14

    @field_validator("horizon")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            msg = "forecast.horizon must be > 0"
            raise ValueError(msg)
        return v


class TimesFMParams(_Base):
    """TimesFM 2.5 (univariate v1). ``context_len`` must be a multiple of 32."""

    type: Literal["timesfm"]
    checkpoint: str = "google/timesfm-2.5-200m-pytorch"
    max_context: int = 1024
    context_len: int = 512
    use_covariates: bool = False
    quantiles: bool = True

    @field_validator("context_len")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0:
            msg = "timesfm.context_len must be a multiple of 32"
            raise ValueError(msg)
        return v


class BQMLParams(_Base):
    """BQML ARIMA model. XREG uses only the available-at-forecast covariate subset."""

    type: Literal["bqml"]
    model_type: Literal["ARIMA_PLUS", "ARIMA_PLUS_XREG"] = "ARIMA_PLUS_XREG"
    use_covariates: bool = True


class AutoMLParams(_Base):
    """Vertex AI / GEAP AutoML forecasting params."""

    type: Literal["automl"]
    optimization_objective: str = "minimize-rmse"
    context_window: int = 28


ModelParams = Annotated[
    TimesFMParams | BQMLParams | AutoMLParams,
    Field(discriminator="type"),
]


class ModelConfig(_Base):
    """A single model entry in the experiment's model list."""

    name: str
    enabled: bool = True
    params: ModelParams


class ExecutionConfig(_Base):
    """Where the experiment runs."""

    target: Literal["local_cpu", "local_gpu", "cloud_pipeline"] = "local_cpu"
    experiment_name: str


class ExperimentConfig(_Base):
    """Root experiment config — the typed contract consumed across geapTimes."""

    project: ProjectConfig
    data: DataConfig = Field(default_factory=DataConfig)
    forecast: ForecastSettings = Field(default_factory=ForecastSettings)
    models: list[ModelConfig]
    execution: ExecutionConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load YAML, substitute ``${ENV}`` references in values, and validate."""
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return cls.model_validate(_expand_env(data))


def _expand_env(obj: object) -> object:
    """Recursively replace ``${VAR}`` in string *values*, raising if any are unset.

    Operates on the parsed structure (not raw text) so YAML comments are never touched.
    """
    if isinstance(obj, dict):
        return {key: _expand_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(_sub_env, obj)
    return obj


def _sub_env(match: re.Match[str]) -> str:
    name = match.group(1)
    value = os.environ.get(name)
    if value is None:
        msg = f"Environment variable '{name}' referenced in config is not set"
        raise ValueError(msg)
    return value
