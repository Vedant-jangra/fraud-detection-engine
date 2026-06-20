import asyncio
import asyncpg
import pandas as pd
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export_data(output_path: str):
    db_url = os.environ.get(
        "JDBC_URL", "postgresql://postgres:fraud_engine_secret@localhost:5432/fraud_db"
    )
    db_url = db_url.replace("jdbc:", "")

    logger.info(f"Connecting to {db_url}")
    conn = await asyncpg.connect(db_url)

    # Note: the exact join depends on the schema in 001_init_schema.sql
    # Usually velocity_features are aggregated by time window.
    # Let's pull the last 30 days of data for retraining.
    query = """
    SELECT 
        t.transaction_id,
        t.user_id,
        t.timestamp,
        t.amount,
        t.merchant_cat,
        t.country,
        t.is_fraud,
        v.txn_count_10m,
        v.total_amount_10m,
        v.unique_merchants_10m,
        v.unique_countries_10m
    FROM transactions t
    LEFT JOIN LATERAL (
        SELECT txn_count_10m, total_amount_10m, unique_merchants_10m, unique_countries_10m
        FROM velocity_features
        WHERE user_id = t.user_id AND window_start <= t.timestamp
        ORDER BY window_start DESC
        LIMIT 1
    ) v ON true
    WHERE t.timestamp >= NOW() - INTERVAL '30 days'
    ORDER BY t.timestamp DESC;
    """

    logger.info("Executing extraction query...")
    records = await conn.fetch(query)

    if not records:
        logger.warning("No records found in the last 30 days.")
        await conn.close()
        return

    # Convert to DataFrame
    columns = [
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "merchant_cat",
        "country",
        "is_fraud",
        "txn_count_10m",
        "total_amount_10m",
        "unique_merchants_10m",
        "unique_countries_10m",
    ]
    df = pd.DataFrame(records, columns=columns)

    # Fill null velocity features with 0
    df = df.fillna(0)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} rows to {output_path}")

    await conn.close()


if __name__ == "__main__":
    output_csv = "data/seeds/retrain_dataset.csv"
    asyncio.run(export_data(output_csv))
