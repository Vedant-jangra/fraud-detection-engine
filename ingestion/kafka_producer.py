import json
import time
import random
import logging
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generators.transaction_simulator import generate_transaction

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_NAME = os.environ.get("KAFKA_TOPIC", "transactions")
DEFAULT_TPS = int(os.environ.get("PRODUCER_TPS", "500"))
DEFAULT_DURATION_SECONDS = int(os.environ.get("PRODUCER_DURATION_SECONDS", "60"))
KAFKA_CONFIG = {
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "client.id": "fraud-producer",
    "linger.ms": 5,
    "batch.size": 16384,
    "compression.type": "snappy",
    "acks": "1",
    "retries": 3,
    "retry.backoff.ms": 100,
}


def ensure_topic_exists(bootstrap_servers: str, topic: str, num_partitions: int = 3):
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    existing = admin.list_topics(timeout=10).topics
    if topic not in existing:
        new_topic = NewTopic(topic, num_partitions=num_partitions, replication_factor=1)
        futures = admin.create_topics([new_topic])
        for t, future in futures.items():
            try:
                future.result()
                logger.info(f"Topic {t} created with {num_partitions} partitions")
            except Exception as e:
                logger.error(f"Failed to create topic {t}: {e}")
    else:
        logger.info(f"Topic {topic} already exists")


def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed for key {msg.key()}: {err}")


def stream_transactions(tps: int = 500, duration_seconds: int = 600):
    if tps <= 0:
        raise ValueError("tps must be greater than 0")

    ensure_topic_exists(KAFKA_BOOTSTRAP, TOPIC_NAME, num_partitions=3)
    producer = Producer(KAFKA_CONFIG)
    user_pool = [f"user_{i:05d}" for i in range(10_000)]
    interval = 1.0 / tps
    start = time.monotonic()
    count = 0
    fraud_count = 0
    last_log_time = time.monotonic()

    logger.info(f"Starting stream: {tps} TPS for {duration_seconds}s")

    while time.monotonic() - start < duration_seconds:
        is_fraud = random.random() < 0.003
        user_id = random.choice(user_pool)
        txn = generate_transaction(user_id, is_fraud)

        producer.produce(
            topic=TOPIC_NAME,
            key=txn["user_id"].encode(),
            value=json.dumps(txn).encode(),
            callback=delivery_report,
        )

        producer.poll(0)

        count += 1
        if is_fraud:
            fraud_count += 1

        if time.monotonic() - last_log_time >= 10.0:
            elapsed = time.monotonic() - start
            actual_tps = count / elapsed
            logger.info(
                f"Elapsed: {elapsed:.0f}s | "
                f"Sent: {count:,} | "
                f"Fraud: {fraud_count} | "
                f"Actual TPS: {actual_tps:.1f}"
            )
            last_log_time = time.monotonic()

        time.sleep(interval)

    producer.flush()
    fraud_rate = fraud_count / count if count else 0
    logger.info(
        f"Stream complete | "
        f"Total: {count:,} | "
        f"Fraud: {fraud_count} | "
        f"Fraud rate: {fraud_rate:.3%}"
    )


if __name__ == "__main__":
    stream_transactions(tps=DEFAULT_TPS, duration_seconds=DEFAULT_DURATION_SECONDS)
