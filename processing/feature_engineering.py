from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


TIMESTAMP_FORMAT = "yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]X"
WATERMARK_DELAY = "2 minutes"
VELOCITY_WINDOW = "10 minutes"
VELOCITY_SLIDE = "1 minute"

TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("merchant_cat", StringType(), True),
        StructField("country", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("ip_address", StringType(), True),
        StructField("is_fraud", IntegerType(), True),
    ]
)

VELOCITY_FEATURE_COLUMNS = [
    "user_id",
    "window_start",
    "window_end",
    "txn_count_10m",
    "total_amount_10m",
    "unique_merchants_10m",
    "unique_countries_10m",
]


def compute_velocity_features(transactions):
    return (
        transactions.withWatermark("event_ts", WATERMARK_DELAY)
        .groupBy(
            F.window("event_ts", VELOCITY_WINDOW, VELOCITY_SLIDE), F.col("user_id")
        )
        .agg(
            F.count("*").alias("txn_count_10m"),
            F.sum("amount").alias("total_amount_10m"),
            F.approx_count_distinct("merchant_cat").alias("unique_merchants_10m"),
            F.approx_count_distinct("country").alias("unique_countries_10m"),
        )
        .select(
            F.col("user_id"),
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("txn_count_10m"),
            F.col("total_amount_10m"),
            F.col("unique_merchants_10m"),
            F.col("unique_countries_10m"),
        )
    )
