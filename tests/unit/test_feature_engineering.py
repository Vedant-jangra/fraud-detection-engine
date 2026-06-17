from processing.feature_engineering import (
    TIMESTAMP_FORMAT,
    TRANSACTION_SCHEMA,
    VELOCITY_FEATURE_COLUMNS,
    VELOCITY_SLIDE,
    VELOCITY_WINDOW,
    WATERMARK_DELAY,
)


def test_transaction_schema_matches_kafka_payload_contract():
    field_names = [field.name for field in TRANSACTION_SCHEMA.fields]

    assert field_names == [
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "merchant_id",
        "merchant_cat",
        "country",
        "device_id",
        "ip_address",
        "is_fraud",
    ]


def test_velocity_feature_columns_match_database_table():
    assert VELOCITY_FEATURE_COLUMNS == [
        "user_id",
        "window_start",
        "window_end",
        "txn_count_10m",
        "total_amount_10m",
        "unique_merchants_10m",
        "unique_countries_10m",
    ]


def test_streaming_window_constants_are_expected_values():
    assert TIMESTAMP_FORMAT == "yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]X"
    assert WATERMARK_DELAY == "2 minutes"
    assert VELOCITY_WINDOW == "10 minutes"
    assert VELOCITY_SLIDE == "1 minute"
