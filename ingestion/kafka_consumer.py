import json
import logging
import os
from collections.abc import Callable
from typing import Any

from confluent_kafka import Consumer, KafkaException, Message


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_NAME = os.environ.get("KAFKA_TOPIC", "transactions")
CONSUMER_GROUP = os.environ.get("KAFKA_CONSUMER_GROUP", "fraud-debug-consumer")

KAFKA_CONFIG = {
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": CONSUMER_GROUP,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}

REQUIRED_TRANSACTION_FIELDS = {
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
}


def decode_transaction_message(message: Message) -> dict[str, Any]:
    raw_value = message.value()
    if raw_value is None:
        raise ValueError("Kafka message value is empty")

    try:
        payload = json.loads(raw_value.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("Kafka message is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Kafka message is not valid JSON") from exc

    missing_fields = REQUIRED_TRANSACTION_FIELDS - payload.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Kafka transaction is missing required fields: {missing}")

    return payload


def log_transaction_summary(transaction: dict[str, Any]) -> None:
    logger.info(
        "transaction_id=%s user_id=%s amount=%.2f country=%s fraud=%s",
        transaction["transaction_id"],
        transaction["user_id"],
        float(transaction["amount"]),
        transaction["country"],
        transaction["is_fraud"],
    )


def consume_transactions(
    max_messages: int | None = None,
    poll_timeout_seconds: float = 1.0,
    message_handler: Callable[[dict[str, Any]], None] = log_transaction_summary,
    consumer: Consumer | None = None,
) -> int:
    if max_messages is not None and max_messages <= 0:
        raise ValueError("max_messages must be greater than 0 when provided")
    if poll_timeout_seconds <= 0:
        raise ValueError("poll_timeout_seconds must be greater than 0")

    kafka_consumer = consumer or Consumer(KAFKA_CONFIG)
    consumed_count = 0

    try:
        kafka_consumer.subscribe([TOPIC_NAME])
        logger.info("Subscribed to Kafka topic '%s'", TOPIC_NAME)

        while max_messages is None or consumed_count < max_messages:
            message = kafka_consumer.poll(poll_timeout_seconds)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())

            transaction = decode_transaction_message(message)
            message_handler(transaction)
            kafka_consumer.commit(message=message, asynchronous=False)
            consumed_count += 1
    except KeyboardInterrupt:
        logger.info("Stopping Kafka consumer after keyboard interrupt")
    finally:
        kafka_consumer.close()

    return consumed_count


if __name__ == "__main__":
    consume_transactions()
