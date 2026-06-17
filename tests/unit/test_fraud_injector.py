from datetime import datetime, timezone

import pytest

from data.generators.fraud_injector import (
    build_fraud_dataset,
    generate_account_takeover_attack,
    generate_bust_out_attack,
    generate_card_testing_attack,
)


def test_card_testing_attack_creates_high_velocity_fraud_records():
    records = generate_card_testing_attack(
        user_id="user_card_test",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        attempts=3,
        interval_seconds=2,
    )

    assert len(records) == 3
    assert {record["is_fraud"] for record in records} == {1}
    assert {record["fraud_pattern"] for record in records} == {"card_testing"}
    assert {record["user_id"] for record in records} == {"user_card_test"}
    assert records[0]["timestamp"] < records[1]["timestamp"] < records[2]["timestamp"]


def test_account_takeover_attack_uses_foreign_country_and_large_amounts():
    records = generate_account_takeover_attack(
        user_id="user_ato",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        transactions=4,
    )

    assert len(records) == 4
    assert all(record["amount"] >= 5_000 for record in records)
    assert all(record["country"] in {"US", "SG", "AE"} for record in records)
    assert {record["fraud_pattern"] for record in records} == {"account_takeover"}


def test_bust_out_attack_contains_warmup_and_fraud_burst():
    records = generate_bust_out_attack(
        user_id="user_bust_out",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        warmup_transactions=2,
        burst_transactions=3,
    )

    assert len(records) == 5
    assert [record["is_fraud"] for record in records] == [0, 0, 1, 1, 1]
    assert records[0]["fraud_pattern"] == "bust_out_warmup"
    assert records[-1]["fraud_pattern"] == "bust_out"


def test_build_fraud_dataset_mixes_normal_and_attack_records():
    dataset = build_fraud_dataset(
        normal_transactions=10,
        attack_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert len(dataset) == 95
    assert dataset["is_fraud"].sum() == 73
    assert {"card_testing", "account_takeover", "bust_out"}.issubset(
        set(dataset["fraud_pattern"].dropna())
    )


def test_card_testing_rejects_invalid_attempt_count():
    with pytest.raises(ValueError, match="attempts must be greater than 0"):
        generate_card_testing_attack(
            user_id="bad_user",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            attempts=0,
        )
