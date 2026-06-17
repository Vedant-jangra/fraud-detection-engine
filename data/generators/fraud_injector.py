from datetime import datetime, timedelta, timezone
import random

import pandas as pd

from data.generators.transaction_simulator import generate_transaction


MERCHANT_CATEGORIES = [
    "grocery",
    "travel",
    "electronics",
    "restaurant",
    "utility",
    "crypto",
]


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_card_testing_attack(
    user_id: str,
    start_time: datetime,
    attempts: int = 60,
    interval_seconds: int = 3,
) -> list[dict]:
    if attempts <= 0:
        raise ValueError("attempts must be greater than 0")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    device_id = f"ct_device_{random.randint(10000, 99999)}"
    ip_address = f"203.0.113.{random.randint(1, 254)}"
    records = []

    for attempt in range(attempts):
        transaction = generate_transaction(user_id=user_id, is_fraud=True)
        transaction.update(
            {
                "timestamp": _format_timestamp(
                    start_time + timedelta(seconds=attempt * interval_seconds)
                ),
                "amount": round(random.uniform(1, 50), 2),
                "merchant_cat": random.choice(MERCHANT_CATEGORIES),
                "country": "IN",
                "device_id": device_id,
                "ip_address": ip_address,
                "fraud_pattern": "card_testing",
            }
        )
        records.append(transaction)

    return records


def generate_account_takeover_attack(
    user_id: str,
    start_time: datetime,
    transactions: int = 8,
    interval_minutes: int = 2,
) -> list[dict]:
    if transactions <= 0:
        raise ValueError("transactions must be greater than 0")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than 0")

    compromised_device = f"ato_device_{random.randint(10000, 99999)}"
    records = []

    for index in range(transactions):
        transaction = generate_transaction(user_id=user_id, is_fraud=True)
        transaction.update(
            {
                "timestamp": _format_timestamp(
                    start_time + timedelta(minutes=index * interval_minutes)
                ),
                "amount": round(random.uniform(5_000, 50_000), 2),
                "merchant_cat": random.choice(["electronics", "crypto", "travel"]),
                "country": random.choice(["US", "SG", "AE"]),
                "device_id": compromised_device,
                "ip_address": f"198.51.100.{random.randint(1, 254)}",
                "fraud_pattern": "account_takeover",
            }
        )
        records.append(transaction)

    return records


def generate_bust_out_attack(
    user_id: str,
    start_time: datetime,
    warmup_transactions: int = 12,
    burst_transactions: int = 5,
) -> list[dict]:
    if warmup_transactions < 0:
        raise ValueError("warmup_transactions cannot be negative")
    if burst_transactions <= 0:
        raise ValueError("burst_transactions must be greater than 0")

    records = []

    for index in range(warmup_transactions):
        transaction = generate_transaction(user_id=user_id, is_fraud=False)
        transaction.update(
            {
                "timestamp": _format_timestamp(start_time + timedelta(days=index)),
                "amount": round(random.uniform(200, 1_500), 2),
                "fraud_pattern": "bust_out_warmup",
            }
        )
        records.append(transaction)

    burst_start = start_time + timedelta(days=warmup_transactions)
    for index in range(burst_transactions):
        transaction = generate_transaction(user_id=user_id, is_fraud=True)
        transaction.update(
            {
                "timestamp": _format_timestamp(
                    burst_start + timedelta(minutes=index * 5)
                ),
                "amount": round(random.uniform(10_000, 75_000), 2),
                "merchant_cat": random.choice(["electronics", "crypto", "travel"]),
                "fraud_pattern": "bust_out",
            }
        )
        records.append(transaction)

    return records


def build_fraud_dataset(
    normal_transactions: int = 1_000,
    attack_start_time: datetime | None = None,
) -> pd.DataFrame:
    if normal_transactions < 0:
        raise ValueError("normal_transactions cannot be negative")

    start_time = attack_start_time or datetime.now(timezone.utc)
    normal_records = [
        generate_transaction(user_id=f"user_{index:05d}", is_fraud=False)
        for index in range(normal_transactions)
    ]
    attack_records = (
        generate_card_testing_attack("fraud_card_testing_001", start_time)
        + generate_account_takeover_attack(
            "fraud_ato_001", start_time + timedelta(hours=1)
        )
        + generate_bust_out_attack(
            "fraud_bust_out_001", start_time + timedelta(hours=2)
        )
    )

    dataset = pd.DataFrame(normal_records + attack_records)
    return dataset.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = build_fraud_dataset(normal_transactions=10_000)
    df.to_csv("data/seeds/transactions_with_fraud_patterns.csv", index=False)
    print(f"Saved {len(df):,} transactions with {df.is_fraud.sum():,} fraud labels")
