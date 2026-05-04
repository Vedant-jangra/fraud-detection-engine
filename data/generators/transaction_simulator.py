from faker import Faker
from datetime import datetime, timezone
import random
import uuid

import numpy as np
import pandas as pd

fake = Faker("en_IN")
rng = np.random.default_rng(seed=42)

FRAUD_ARCHETYPES = {
    "card_testing": {"amount_range": (1, 50), "velocity": "high", "weight": 0.35},
    "account_takeover": {
        "amount_range": (5000, 50000),
        "velocity": "med",
        "weight": 0.40,
    },
    "bust_out": {"amount_range": (1000, 10000), "velocity": "low", "weight": 0.25},
}


def generate_transaction(user_id: str, is_fraud: bool = False) -> dict:
    base_amount = float(rng.lognormal(mean=6.5, sigma=1.2))
    if is_fraud:
        archetype = random.choices(
            list(FRAUD_ARCHETYPES.keys()),
            weights=[v["weight"] for v in FRAUD_ARCHETYPES.values()],
        )[0]
        lo, hi = FRAUD_ARCHETYPES[archetype]["amount_range"]
        amount = round(random.uniform(lo, hi), 2)
    else:
        amount = round(base_amount, 2)
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "amount": amount,
        "merchant_id": fake.bothify(text="MER-####??"),
        "merchant_cat": random.choice(
            ["grocery", "travel", "electronics", "restaurant", "utility", "crypto"]
        ),
        "country": random.choices(
            ["IN", "US", "SG", "AE"], weights=[0.70, 0.15, 0.10, 0.05]
        )[0],
        "device_id": fake.md5()[:16],
        "ip_address": fake.ipv4(),
        "is_fraud": int(is_fraud),
    }


def generate_dataset(
    n_transactions: int = 100_000, fraud_rate: float = 0.003
) -> pd.DataFrame:
    user_pool = [fake.uuid4() for _ in range(10_000)]
    fraud_count = int(n_transactions * fraud_rate)
    legit_count = n_transactions - fraud_count
    records = [
        generate_transaction(random.choice(user_pool), False)
        for _ in range(legit_count)
    ] + [
        generate_transaction(random.choice(user_pool), True) for _ in range(fraud_count)
    ]
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(df):,} transactions | Fraud rate: {df.is_fraud.mean():.3%}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.shape)
    print(df.is_fraud.value_counts())
    print(df.amount.describe())
    df.to_csv("data/seeds/transactions.csv", index=False)
    print("Saved to data/seeds/transactions.csv")
