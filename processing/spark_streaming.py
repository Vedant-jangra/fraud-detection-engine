import os
from typing import Any

from pyspark.sql import SparkSession

from processing.event_quality import (
    classify_transaction_events,
    select_valid_transaction_events,
)
from processing.feature_engineering import (
    compute_velocity_features,
)


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:29092")
TOPIC_NAME = "transactions"
JDBC_URL = os.environ.get("JDBC_URL", "jdbc:postgresql://timescaledb:5432/fraud_db")
JDBC_PROPS = {
    "user": "postgres",
    "password": "fraud_engine_secret",
    "driver": "org.postgresql.Driver",
}


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

    classified_events = classify_transaction_events(raw_stream)
    valid_events = select_valid_transaction_events(classified_events)
    velocity = compute_velocity_features(valid_events)

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
