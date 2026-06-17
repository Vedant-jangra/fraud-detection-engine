from processing.event_quality import (
    EVENT_QUALITY_COLUMNS,
    REQUIRED_TRANSACTION_COLUMNS,
)


def test_required_transaction_columns_match_stream_schema_contract():
    assert REQUIRED_TRANSACTION_COLUMNS == [
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


def test_event_quality_columns_cover_validation_outputs():
    assert EVENT_QUALITY_COLUMNS == [
        "raw_value",
        "event_ts",
        "is_valid_json",
        "has_required_fields",
        "has_valid_event_ts",
    ]
