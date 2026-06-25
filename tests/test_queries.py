"""String-level tests for the SQL builders (no BigQuery calls)."""

from geaptimes.data.queries import (
    PREPPED_TABLE_NAME,
    SOURCE_TABLE_NAME,
    build_prepped_query,
    build_source_query,
    table_id,
)
from geaptimes.schemas import DataConfig, ProjectConfig


def test_table_id() -> None:
    proj = ProjectConfig(id="proj", gcs_bucket="bkt", bq_dataset="ds")
    assert table_id(proj, SOURCE_TABLE_NAME) == "proj.ds.citibike_daily_source"


def test_source_query_core_elements() -> None:
    data = DataConfig()
    sql = build_source_query(data, "proj.ds.citibike_daily_source")
    assert "CREATE OR REPLACE TABLE `proj.ds.citibike_daily_source`" in sql
    assert data.trips_table in sql
    assert "LIMIT 25" in sql
    assert "COUNT(*) AS num_trips" in sql
    # weather join + sentinels + station selection
    assert ".gsod*`" in sql
    assert "NULLIF(temp, 9999.9)" in sql
    assert "stn = '725030'" in sql
    # calendar features + station metadata
    assert "AS day_of_week" in sql
    assert "AS is_weekend" in sql
    assert "capacity, region_id, latitude, longitude" in sql


def test_source_query_gap_fill_toggle() -> None:
    with_fill = build_source_query(DataConfig(gap_fill=True), "p.d.s")
    without_fill = build_source_query(DataConfig(gap_fill=False), "p.d.s")
    assert "GENERATE_DATE_ARRAY" in with_fill
    assert "GENERATE_DATE_ARRAY" not in without_fill


def test_source_query_date_filter() -> None:
    data = DataConfig()
    data.date_range.start = "2015-01-01"
    data.date_range.end = "2017-12-31"
    sql = build_source_query(data, "p.d.s")
    assert "DATE('2015-01-01')" in sql
    assert "DATE('2017-12-31')" in sql
    assert "BETWEEN '2015' AND '2017'" in sql  # GSOD year window follows the range


def test_prepped_query_splits() -> None:
    data = DataConfig()  # test_length=14, validate_length=14
    sql = build_prepped_query(data, "p.d.citibike_daily_source", f"p.d.{PREPPED_TABLE_NAME}")
    assert "INTERVAL 14 DAY) THEN 'TEST'" in sql
    assert "INTERVAL 28 DAY) THEN 'VALIDATE'" in sql
    assert "ELSE 'TRAIN'" in sql
    assert "AS splits" in sql
