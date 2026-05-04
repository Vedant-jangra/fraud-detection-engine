import os
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, DoubleType, IntegerType


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:29092")
TOPIC_NAME = "transactions"
JDBC_URL = os.environ.get("JDBC_URL", "jdbc:postgresql://timescaledb:5432/fraud_db")
JDBC_PROPS = {
    "user": "postgres",
    "password": "fraud_engine_secret",
    "driver": "org.postgresql.Driver",
}

TXN_SCHEMA = StructType(
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


def create_spark_session() -> SparkSession:
    builder: Any = SparkSession.builder
    return builder.appName("FraudFeatureEngine").getOrCreate()


def read_kafka_stream(spark: SparkSession):
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC_NAME)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 5000)
        .option("kafka.group.id", "spark-fraud-engine")
        .load()
    )


def parse_transactions(raw_stream):
    parsed = raw_stream.select(
        F.from_json(F.col("value").cast("string"), TXN_SCHEMA).alias("data")
    ).select("data.*")

    return parsed.withColumn(
        "event_ts", F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]X")
    )


def compute_velocity_features(parsed):
    return (
        parsed.withWatermark("event_ts", "2 minutes")
        .groupBy(F.window("event_ts", "10 minutes", "1 minute"), F.col("user_id"))
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


def write_batch(df, epoch_id):
    count = df.count()
    if count > 0:
        df.write.jdbc(
            url=JDBC_URL,
            table="velocity_features",
            mode="append",
            properties=JDBC_PROPS,
        )
        print(f"Batch {epoch_id}: wrote {count} velocity feature rows")
    else:
        print(f"Batch {epoch_id}: no data to write")


def run_streaming_job():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    print("Spark session created successfully")

    raw_stream = read_kafka_stream(spark)
    print("Kafka stream connected")

    parsed = parse_transactions(raw_stream)
    velocity = compute_velocity_features(parsed)

    query = (
        velocity.writeStream.foreachBatch(write_batch)
        .trigger(processingTime="10 seconds")
        .outputMode("update")
        .start()
    )

    print("Streaming query started. Waiting for data...")
    print("Press Ctrl+C to stop")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("Stopping streaming job...")
        query.stop()
        spark.stop()
        print("Stopped cleanly")


if __name__ == "__main__":
    run_streaming_job()
