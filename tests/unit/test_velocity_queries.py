from pathlib import Path


QUERY_FILE = Path("feature_store/velocity_queries.sql")


def test_velocity_queries_cover_expected_fraud_patterns():
    sql = QUERY_FILE.read_text(encoding="utf-8")

    assert "card_testing" in sql
    assert "account_takeover" in sql
    assert "bust_out" in sql
    assert "velocity_features" in sql


def test_velocity_queries_reference_existing_schema_columns():
    sql = QUERY_FILE.read_text(encoding="utf-8")

    for column_name in [
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "merchant_id",
        "merchant_cat",
        "country",
        "device_id",
        "ip_address",
        "txn_count_10m",
        "total_amount_10m",
        "unique_merchants_10m",
        "unique_countries_10m",
    ]:
        assert column_name in sql


def test_velocity_queries_avoid_unsupported_distinct_window_aggregate():
    sql = QUERY_FILE.read_text(encoding="utf-8").lower()

    assert "count(distinct" in sql
    assert "count(distinct merchant_id) over" not in sql
    assert "count(distinct country) over" not in sql
