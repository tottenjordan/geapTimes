"""String-level tests for the SQL builders (no BigQuery calls)."""

from geaptimes.data.queries import (
    build_infer_query,
    build_prepped_query,
    build_source_query,
    build_train_query,
    table_id,
)
from geaptimes.schemas import DataConfig, ProjectConfig


def test_table_id() -> None:
    proj = ProjectConfig(id="proj", gcs_bucket="bkt", bq_dataset="ds")
    assert table_id(proj, DataConfig().source_table_name) == "proj.ds.citibike_daily_source"


def test_source_query_core_elements() -> None:
    data = DataConfig()
    sql = build_source_query(data, "proj.ds.citibike_daily_source")
    assert "CREATE OR REPLACE TABLE `proj.ds.citibike_daily_source`" in sql
    assert 'OPTIONS(labels=[("solution", "geaptimes")])' in sql  # required GCP label
    assert data.trips_table in sql
    assert "LIMIT 25" in sql
    assert "!= ''" in sql  # exclude the empty-string (missing) station name
    assert "COUNT(*) AS num_trips" in sql
    # gender/usertype are STRING in this table (not numeric codes)
    assert "gender = 'male'" in sql
    # weather join + sentinels + station selection
    assert ".gsod*`" in sql
    assert "NULLIF(temp, 9999.9)" in sql
    assert "stn = '725030'" in sql
    # calendar features + station metadata
    assert "AS day_of_week" in sql
    assert "AS is_weekend" in sql
    assert "capacity, region_id, latitude, longitude" in sql
    # metadata joins on station_id (more stable than name)
    assert "station_ids AS" in sql
    assert "CAST(m.station_id AS STRING)" in sql


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
    sql = build_prepped_query(data, "p.d.citibike_daily_source", f"p.d.{data.prepped_table_name}")
    assert "INTERVAL 14 DAY) THEN 'TEST'" in sql
    assert "INTERVAL 28 DAY) THEN 'VALIDATE'" in sql
    assert "ELSE 'TRAIN'" in sql
    assert "AS splits" in sql
    assert 'OPTIONS(labels=[("solution", "geaptimes")])' in sql  # required GCP label


def test_train_query_excludes_test() -> None:
    sql = build_train_query(DataConfig(), "p.d.prepped", "p.d.train")
    assert "CREATE OR REPLACE TABLE `p.d.train`" in sql
    assert 'OPTIONS(labels=[("solution", "geaptimes")])' in sql
    assert "FROM `p.d.prepped`" in sql
    assert "WHERE splits != 'TEST'" in sql


def test_infer_query_is_test_window_with_null_target() -> None:
    data = DataConfig()  # target_column = num_trips
    sql = build_infer_query(data, "p.d.prepped", "p.d.infer")
    assert "CREATE OR REPLACE TABLE `p.d.infer`" in sql
    assert 'OPTIONS(labels=[("solution", "geaptimes")])' in sql
    assert "WHERE splits = 'TEST'" in sql
    # target is nulled so the row schema matches a real inference request
    assert "REPLACE (CAST(NULL AS INT64) AS num_trips)" in sql
