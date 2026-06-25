"""Forecaster abstraction shared by every backend.

Each backend (TimesFM, BQML, AutoML) has a radically different execution model, but they all
implement the same :class:`Forecaster` interface and return the same :class:`ForecastResult`
shape so that downstream experiment tracking (Stage 3) and evaluation (Stage 5) are
backend-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from geaptimes.schemas import ExperimentConfig, ModelConfig

# Standardized quantile levels emitted by every backend.
QUANTILES: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

# Canonical prediction-frame column names (single source of truth for all backends + tests).
SERIES_COLUMN = "series"
DATE_COLUMN = "date"
HORIZON_STEP_COLUMN = "horizon_step"
FORECAST_COLUMN = "forecast"
MODEL_COLUMN = "model"


def quantile_column(level: float) -> str:
    """Map a quantile level (e.g. ``0.1``) to its column name (e.g. ``"q10"``)."""
    return f"q{round(level * 100)}"


# Ordered prediction columns: keys + point forecast + quantiles + model tag.
QUANTILE_COLUMNS: list[str] = [quantile_column(q) for q in QUANTILES]
PREDICTION_COLUMNS: list[str] = [
    SERIES_COLUMN,
    DATE_COLUMN,
    HORIZON_STEP_COLUMN,
    FORECAST_COLUMN,
    *QUANTILE_COLUMNS,
    MODEL_COLUMN,
]


@dataclass(frozen=True)
class ForecastResult:
    """Uniform forecast output.

    ``predictions`` is a long-format frame with :data:`PREDICTION_COLUMNS`; ``metadata`` carries
    backend-specific artifacts (device/checkpoint for TimesFM, model id for BQML, job display
    name for AutoML) for traceability.
    """

    predictions: "pd.DataFrame"
    metadata: dict[str, Any] = field(default_factory=dict)


class Forecaster(ABC):
    """Common interface for a single configured model.

    Subclasses receive the model's own :class:`~geaptimes.schemas.ModelConfig` plus the full
    :class:`~geaptimes.schemas.ExperimentConfig` (for project/data/forecast context).
    """

    def __init__(self, model_cfg: "ModelConfig", cfg: "ExperimentConfig") -> None:
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.name = model_cfg.name

    @abstractmethod
    def fit(self) -> None:
        """Train (or no-op for zero-shot models like TimesFM)."""

    @abstractmethod
    def predict(self) -> ForecastResult:
        """Produce a forecast over the configured horizon."""
