"""Typed configuration schema for geapTimes experiments.

Experiments are defined in YAML and validated into these Pydantic v2 models. Application code
consumes the validated :class:`ExperimentConfig` (and its members) — never raw YAML.

Per-model parameters use a discriminated union on ``params.type`` so each model carries only its
own, type-checked fields.
"""

import os
import re
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


class _Base(BaseModel):
    """Base model: forbid unknown keys so config typos fail loudly."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ProjectConfig(_Base):
    """GCP project / storage targets.

    ``region`` is the Vertex/GCS region; ``bq_location`` is the BigQuery dataset location, which
    must colocate with the source data. The ``bigquery-public-data`` Citibike/GSOD tables live in
    the ``US`` multi-region, so derived datasets default to ``US`` (not a single region).
    """

    id: str
    region: str = "us-central1"
    bq_location: str = "US"
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
    # Output table base names (within project.bq_dataset). geaptimes.naming appends a slug:
    # the shared `source` gets the data slug; prepped/train/infer get the full config slug.
    source_table_name: str = "citibike_daily_source"
    prepped_table_name: str = "citibike_daily_prepped"
    train_table_name: str = "citibike_daily_train"
    infer_table_name: str = "citibike_daily_infer"
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
    # The framework always emits quantile forecasts (q10..q90), and Vertex requires
    # "minimize-quantile-loss" whenever quantiles are requested -- any other objective is rejected.
    optimization_objective: str = "minimize-quantile-loss"
    context_window: int = 28
    # Training budget in milli-node-hours (1000 = 1 node-hour). Bounds AutoML's billable spend;
    # the wrapper is gated by default (model disabled) and only runs when explicitly enabled.
    budget_milli_node_hours: int = 1000

    @field_validator("budget_milli_node_hours")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            msg = "automl.budget_milli_node_hours must be > 0"
            raise ValueError(msg)
        return v


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


class DOEConfig(_Base):
    """Design-of-Experiments sweep.

    ``axes`` maps a dotted config path (e.g. ``"forecast.horizon"``) to the list of values to
    sweep. :func:`geaptimes.experiment.doe.expand` takes the cross-product of all axes, applying
    each combination as an override on the base config. Empty ``axes`` means a single run (the
    base config unchanged).
    """

    axes: dict[str, list[Any]] = Field(default_factory=dict)


class ArtifactRegistryConfig(_Base):
    """Artifact Registry coordinates for the geapTimes runtime container image.

    The image serves double duty: the TimesFM custom serving container and the base image for the
    KFP pipeline components. ``location`` defaults to ``project.region`` when unset.
    """

    repo: str = "geaptimes"
    image_name: str = "geaptimes-runtime"
    image_tag: str = "latest"
    location: str | None = None

    def image_uri(self, *, project_id: str, region: str) -> str:
        """Resolve the fully-qualified Artifact Registry image URI."""
        location = self.location or region
        return (
            f"{location}-docker.pkg.dev/{project_id}/{self.repo}/{self.image_name}:{self.image_tag}"
        )


class ServingConfig(_Base):
    """How the custom-container TimesFM model is served in the pipeline.

    ``mode`` selects an online endpoint (default) or a batch-prediction job. ``keep_deployed``
    is the cost switch: ``False`` (default) tears the endpoint down after artifacts are saved;
    ``True`` leaves it running.
    """

    mode: Literal["endpoint", "batch"] = "endpoint"
    keep_deployed: bool = False
    machine_type: str = "n1-standard-4"
    min_replica_count: int = 1
    max_replica_count: int = 1
    batch_machine_type: str = "n1-standard-4"
    batch_starting_replica_count: int = 1
    batch_max_replica_count: int = 1

    @field_validator(
        "min_replica_count",
        "max_replica_count",
        "batch_starting_replica_count",
        "batch_max_replica_count",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            msg = "serving replica counts must be > 0"
            raise ValueError(msg)
        return v


class PipelineConfig(_Base):
    """Vertex AI managed-pipeline (KFP) settings for cloud orchestration."""

    pipeline_root: str | None = None
    image: ArtifactRegistryConfig = Field(default_factory=ArtifactRegistryConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    component_machine_type: str = "e2-standard-4"
    enable_caching: bool = True

    def resolved_pipeline_root(self, *, gcs_bucket: str, experiment_name: str) -> str:
        """Resolve the pipeline root, defaulting to a per-experiment path in the GCS bucket."""
        if self.pipeline_root:
            return self.pipeline_root
        return f"gs://{gcs_bucket}/pipeline_root/{experiment_name}"


class ExperimentConfig(_Base):
    """Root experiment config — the typed contract consumed across geapTimes."""

    project: ProjectConfig
    data: DataConfig = Field(default_factory=DataConfig)
    forecast: ForecastSettings = Field(default_factory=ForecastSettings)
    models: list[ModelConfig]
    execution: ExecutionConfig
    doe: DOEConfig = Field(default_factory=DOEConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

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
