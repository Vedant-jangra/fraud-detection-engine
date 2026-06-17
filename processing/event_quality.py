from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from processing.feature_engineering import TIMESTAMP_FORMAT, TRANSACTION_SCHEMA


EVENT_QUALITY_COLUMNS = [
    "raw_value",
    "event_ts",
    "is_valid_json",
    "has_required_fields",
    "has_valid_event_ts",
]

REQUIRED_TRANSACTION_COLUMNS = [
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


def classify_transaction_events(raw_stream: DataFrame) -> DataFrame:
    parsed = raw_stream.select(
        F.col("value").cast("string").alias("raw_value"),
        F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("data"),
    )

    required_field_checks = [
        F.col(f"data.{column_name}").isNotNull()
        for column_name in REQUIRED_TRANSACTION_COLUMNS
    ]
    has_required_fields = required_field_checks[0]
    for condition in required_field_checks[1:]:
        has_required_fields = has_required_fields & condition

    return (
        parsed.select("raw_value", "data.*")
        .withColumn("is_valid_json", F.col("transaction_id").isNotNull())
        .withColumn("has_required_fields", has_required_fields)
        .withColumn("event_ts", F.to_timestamp("timestamp", TIMESTAMP_FORMAT))
        .withColumn("has_valid_event_ts", F.col("event_ts").isNotNull())
    )


def select_valid_transaction_events(classified_events: DataFrame) -> DataFrame:
    return classified_events.filter(
        F.col("is_valid_json")
        & F.col("has_required_fields")
        & F.col("has_valid_event_ts")
    )
