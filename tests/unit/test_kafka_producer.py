import pytest

from ingestion.kafka_producer import KAFKA_CONFIG, stream_transactions


def test_kafka_config_uses_bootstrap_servers():
    assert KAFKA_CONFIG["bootstrap.servers"]


def test_stream_transactions_rejects_non_positive_tps():
    with pytest.raises(ValueError, match="tps must be greater than 0"):
        stream_transactions(tps=0, duration_seconds=1)
