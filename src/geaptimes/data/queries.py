"""BigQuery SQL builders for the Citibike + weather forecasting tables.

Mirrors the statmike ``_source`` -> ``_prepped`` pattern:

* :func:`build_source_query` aggregates Citibike trips to a daily, per-station series, joins
  NOAA GSOD weather + station metadata, and adds calendar features.
* :func:`build_prepped_query` adds a ``splits`` column (TRAIN/VALIDATE/TEST) by date range off
  the global max date.

SQL is assembled from trusted :class:`~geaptimes.schemas.DataConfig` values (not user input).
"""

from geaptimes.constants import bq_labels_option, bq_table_options
from geaptimes.schemas import DataConfig, ProjectConfig

# GSOD missing-value sentinels.
_TEMP_MISSING = "9999.9"
_PRCP_MISSING = "99.99"

# Default GSOD year window (Citibike public table is frozen ~2018).
_DEFAULT_START_YEAR = "2013"
_DEFAULT_END_YEAR = "2019"

# Length of the YYYY prefix in an ISO date string.
_YEAR_PREFIX_LEN = 4


def table_id(project: ProjectConfig, name: str) -> str:
    """Fully-qualified ``project.dataset.table`` id."""
    return f"{project.id}.{project.bq_dataset}.{name}"


def _year(date_str: str | None, default: str) -> str:
    """Extract the YYYY prefix from an ISO date, or fall back to ``default``."""
    if date_str and len(date_str) >= _YEAR_PREFIX_LEN and date_str[:_YEAR_PREFIX_LEN].isdigit():
        return date_str[:_YEAR_PREFIX_LEN]
    return default


def build_source_query(
    data: DataConfig, destination: str, *, description: str | None = None
) -> str:
    """Build ``CREATE OR REPLACE TABLE <destination>`` for the daily source table.

    *description* (the config fingerprint hash) is embedded in the table options so the
    self-bootstrapping ``ensure_source`` guard can read it back and skip a rebuild when current.
    """
    series = data.series_column
    time = data.time_column
    target = data.target_column

    # Optional explicit date filter on the trips.
    date_filter = ""
    if data.date_range.start:
        date_filter += f"\n    AND DATE(starttime) >= DATE('{data.date_range.start}')"
    if data.date_range.end:
        date_filter += f"\n    AND DATE(starttime) <= DATE('{data.date_range.end}')"

    y0 = _year(data.date_range.start, _DEFAULT_START_YEAR)
    y1 = _year(data.date_range.end, _DEFAULT_END_YEAR)

    if data.gap_fill:
        # Per-station active-window date spine, zero-filling no-trip days.
        series_cte = f"""bounds AS (
    SELECT {series}, MIN({time}) AS d0, MAX({time}) AS d1
    FROM daily GROUP BY {series}
),
spine AS (
    SELECT b.{series}, d AS {time}
    FROM bounds b, UNNEST(GENERATE_DATE_ARRAY(b.d0, b.d1)) AS d
),
series_tbl AS (
    SELECT
        s.{series}, s.{time},
        IFNULL(d.{target}, 0) AS {target},
        d.avg_tripduration, d.pct_subscriber, d.ratio_gender
    FROM spine s
    LEFT JOIN daily d USING ({series}, {time})
)"""
    else:
        series_cte = "series_tbl AS (SELECT * FROM daily)"

    return f"""CREATE OR REPLACE TABLE `{destination}`
{bq_table_options(description=description)} AS
WITH top_stations AS (
    SELECT {series}
    FROM `{data.trips_table}`
    WHERE {series} IS NOT NULL AND {series} != ''
    GROUP BY {series}
    ORDER BY COUNT(*) DESC
    LIMIT {data.station_filter.top_n}
),
daily AS (
    SELECT
        {series},
        DATE(starttime) AS {time},
        COUNT(*) AS {target},
        AVG(tripduration) AS avg_tripduration,
        AVG(IF(usertype = 'Subscriber', 1.0, 0.0)) AS pct_subscriber,
        SAFE_DIVIDE(COUNTIF(gender = 'male'), NULLIF(COUNTIF(gender IN ('male', 'female')), 0)) AS ratio_gender
    FROM `{data.trips_table}`
    WHERE {series} IN (SELECT {series} FROM top_stations){date_filter}
    GROUP BY {series}, {time}
),
{series_cte},
station_ids AS (
    SELECT {series}, start_station_id
    FROM (
        SELECT {series}, start_station_id,
               ROW_NUMBER() OVER (PARTITION BY {series} ORDER BY COUNT(*) DESC) AS rn
        FROM `{data.trips_table}`
        WHERE {series} IN (SELECT {series} FROM top_stations)
        GROUP BY {series}, start_station_id
    )
    WHERE rn = 1
),
weather AS (
    SELECT
        PARSE_DATE('%Y%m%d', CONCAT(year, mo, da)) AS {time},
        NULLIF(temp, {_TEMP_MISSING}) AS temp,
        NULLIF(prcp, {_PRCP_MISSING}) AS prcp
    FROM `{data.weather_dataset}.gsod*`
    WHERE _TABLE_SUFFIX BETWEEN '{y0}' AND '{y1}'
        AND stn = '{data.weather.station_usaf}'
        AND wban = '{data.weather.station_wban}'
),
meta AS (
    SELECT station_id, capacity, region_id, latitude, longitude
    FROM `{data.stations_table}`
)
SELECT
    t.{series}, t.{time}, t.{target},
    t.avg_tripduration, t.pct_subscriber, t.ratio_gender,
    w.temp, w.prcp,
    EXTRACT(DAYOFWEEK FROM t.{time}) AS day_of_week,
    EXTRACT(MONTH FROM t.{time}) AS month,
    IF(EXTRACT(DAYOFWEEK FROM t.{time}) IN (1, 7), 1, 0) AS is_weekend,
    sid.start_station_id AS station_id,
    m.capacity, m.region_id, m.latitude, m.longitude
FROM series_tbl t
LEFT JOIN weather w USING ({time})
LEFT JOIN station_ids sid USING ({series})
LEFT JOIN meta m ON CAST(m.station_id AS STRING) = CAST(sid.start_station_id AS STRING)
"""


def build_prepped_query(
    data: DataConfig, source: str, destination: str, *, description: str | None = None
) -> str:
    """Build ``CREATE OR REPLACE TABLE <destination>`` adding the ``splits`` column.

    *description* (the config fingerprint hash) is embedded in the table options so the
    self-bootstrapping ``ensure_prepped`` guard can read it back and skip a rebuild when current.
    """
    time = data.time_column
    test = data.splits.test_length
    test_plus_val = data.splits.test_length + data.splits.validate_length
    return f"""CREATE OR REPLACE TABLE `{destination}`
{bq_table_options(description=description)} AS
SELECT *,
    CASE
        WHEN {time} > DATE_SUB((SELECT MAX({time}) FROM `{source}`), INTERVAL {test} DAY) THEN 'TEST'
        WHEN {time} > DATE_SUB((SELECT MAX({time}) FROM `{source}`), INTERVAL {test_plus_val} DAY) THEN 'VALIDATE'
        ELSE 'TRAIN'
    END AS splits
FROM `{source}`
"""


def build_train_query(data: DataConfig, prepped: str, destination: str) -> str:
    """Build the training table: all non-TEST rows (TRAIN + VALIDATE) from the prepped table.

    This is the fitting window backends train on; forecasting ``horizon`` steps past its last
    date lands on the held-out TEST window (see :func:`build_infer_query`).
    """
    _ = data  # signature parity with the other builders; no field needed today.
    return f"""CREATE OR REPLACE TABLE `{destination}`
{bq_labels_option()} AS
SELECT * FROM `{prepped}`
WHERE splits != 'TEST'
"""


def build_infer_query(data: DataConfig, prepped: str, destination: str) -> str:
    """Build the inference table: the TEST window with the target nulled out.

    For this frozen public dataset the held-out forecast horizon *is* the TEST split (the most
    recent ``splits.test_length`` days), so the known-future regressors (temp/prcp/calendar) are
    genuinely available here — unlike true production dates past the table's max, which have no
    weather actuals (deferred to Stage 4). The target is set NULL so the row schema matches a
    real inference request: it is what BQML ``ML.FORECAST`` (XREG future regressors) and AutoML
    batch prediction consume.
    """
    target = data.target_column
    return f"""CREATE OR REPLACE TABLE `{destination}`
{bq_labels_option()} AS
SELECT * REPLACE (CAST(NULL AS INT64) AS {target})
FROM `{prepped}`
WHERE splits = 'TEST'
"""
