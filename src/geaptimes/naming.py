"""Deterministic table naming for geapTimes experiments (single source of truth).

Convention (human-readable tokens):

* shared ``source`` table  -> ``<base>__<data_slug>``   (keyed on data-shape tokens)
* ``prepped``/``train``/``infer`` -> ``<base>__<config_slug>`` (adds split + horizon tokens)

Token slugs intentionally do **not** encode covariate/weather/holiday changes; if those vary
between experiments without other token changes, the table names collide. Disambiguate by using a
distinct ``execution.experiment_name`` and rely on the table description (see
:func:`config_fingerprint`) for full traceability.
"""

import json

from geaptimes.schemas import DataConfig, ExperimentConfig, ForecastSettings


def data_slug(data: DataConfig) -> str:
    """Tokens for the shared source layer (data-shape knobs)."""
    return f"top{data.station_filter.top_n}"


def config_slug(data: DataConfig, forecast: ForecastSettings) -> str:
    """Tokens for config-specific tables: top-N, horizon, test/validate split lengths."""
    return (
        f"top{data.station_filter.top_n}"
        f"_h{forecast.horizon}"
        f"_t{data.splits.test_length}"
        f"_v{data.splits.validate_length}"
    )


def table_names(cfg: ExperimentConfig) -> dict[str, str]:
    """Map roles (source/prepped/train/infer) to their table base names (no project/dataset)."""
    ds = data_slug(cfg.data)
    cs = config_slug(cfg.data, cfg.forecast)
    return {
        "source": f"{cfg.data.source_table_name}__{ds}",
        "prepped": f"{cfg.data.prepped_table_name}__{cs}",
        "train": f"{cfg.data.train_table_name}__{cs}",
        "infer": f"{cfg.data.infer_table_name}__{cs}",
    }


def config_fingerprint(cfg: ExperimentConfig) -> str:
    """Stable JSON fingerprint of the data+forecast config, for table descriptions/traceability."""
    payload = {
        "experiment": cfg.execution.experiment_name,
        "data": cfg.data.model_dump(),
        "forecast": cfg.forecast.model_dump(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
