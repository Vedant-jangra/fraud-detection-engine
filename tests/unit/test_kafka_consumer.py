import json

import pytest

from ingestion.kafka_consumer import (
    REQUIRED_TRANSACTION_FIELDS,
    consume_transactions,
    decode_transaction_message,
)


class FakeMessage:
    def __init__(self, value: bytes | None, error=None):
        self._value = value
        self._error = error

    def value(self):
        return self._value

    def error(self):
        return self._error


class FakeConsumer:
    def __init__(self, messages):
        self.messages = list(messages)
        self.subscribed_topics = []
        self.commit_count = 0
        self.closed = False

    def subscribe(self, topics):
        self.subscribed_topics = topics

    def poll(self, timeout):
        if self.messages:
            return self.messages.pop(0)
        return None

    def commit(self, message, asynchronous):
        self.commit_count += 1

    def close(self):
        self.closed = True


def valid_transaction_payload() -> dict:
    return {
        "transaction_id": "txn_1",
        "user_id": "user_1",
        "timestamp": "2026-01-01T00:00:00Z",
        "amount": 42.5,
        "merchant_id": "MER-1234AB",
        "merchant_cat": "grocery",
        "country": "IN",
        "device_id": "device_1",
        "ip_address": "203.0.113.10",
        "is_fraud": 0,
    }


def encode_payload(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_decode_transaction_message_returns_valid_payload():
    payload = valid_transaction_payload()
    message = FakeMessage(encode_payload(payload))

    assert decode_transaction_message(message) == payload


def test_decode_transaction_message_rejects_invalid_json():
    message = FakeMessage(b"not-json")

    with pytest.raises(ValueError, match="not valid JSON"):
        decode_transaction_message(message)


def test_decode_transaction_message_rejects_missing_fields():
    payload = valid_transaction_payload()
    payload.pop("merchant_id")
    message = FakeMessage(encode_payload(payload))

    with pytest.raises(ValueError, match="merchant_id"):
        decode_transaction_message(message)


def test_required_fields_match_transaction_schema():
    assert REQUIRED_TRANSACTION_FIELDS == set(valid_transaction_payload())


def test_consume_transactions_handles_messages_and_commits():
    payload = valid_transaction_payload()
    consumer = FakeConsumer([FakeMessage(encode_payload(payload))])
    handled = []

    consumed_count = consume_transactions(
        max_messages=1,
        poll_timeout_seconds=0.1,
        message_handler=handled.append,
        consumer=consumer,
    )

    assert consumed_count == 1
    assert handled == [payload]
    assert consumer.subscribed_topics == ["transactions"]
    assert consumer.commit_count == 1
    assert consumer.closed is True


def test_consume_transactions_rejects_invalid_max_messages():
    with pytest.raises(ValueError, match="max_messages"):
        consume_transactions(max_messages=0)
