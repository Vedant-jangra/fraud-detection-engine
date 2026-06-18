from pathlib import Path


QUERY_FILE = Path("feature_store/velocity_queries.sql")


def test_velocity_queries_contain_all_four_ctes():
    sql = QUERY_FILE.read_text(encoding="utf-8")

    assert "recent_activity" in sql
    assert "historical_baseline" in sql
    assert "window_velocity" in sql
    assert "velocity_ratio" in sql


def test_velocity_queries_reference_expected_feature_columns():
    sql = QUERY_FILE.read_text(encoding="utf-8")

    for column_name in [
        "user_id",
        "timestamp",
        "amount",
        "merchant_id",
        "country",
        "device_id",
        "txn_count_10m",
        "total_amount_10m",
        "unique_merchants_10m",
        "unique_countries_10m",
    ]:
        assert column_name in sql


def test_velocity_queries_compute_key_fraud_signals():
    sql = QUERY_FILE.read_text(encoding="utf-8")

    assert "velocity_ratio" in sql
    assert "amount_spike_ratio" in sql
    assert "high_velocity_flag" in sql
    assert "amount_spike_flag" in sql
    assert "amount_zscore" in sql
    assert "time_since_last_txn" in sql


def test_velocity_queries_avoid_unsupported_distinct_window_aggregate():
    sql = QUERY_FILE.read_text(encoding="utf-8").lower()

    assert "count(distinct" in sql
    assert "count(distinct merchant_id) over" not in sql
    assert "count(distinct country) over" not in sql
