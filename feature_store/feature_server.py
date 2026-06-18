from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncpg
import os
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:fraud_engine_secret@localhost:5432/fraud_db",
    )
    app.state.db_pool = await asyncpg.create_pool(
        dsn=db_url, min_size=5, max_size=20, command_timeout=5.0
    )
    logger.info("Feature server ready")
    yield
    await app.state.db_pool.close()


app = FastAPI(title="Feature Store Server", version="1.0.0", lifespan=lifespan)


@app.get("/features/velocity/{user_id}")
async def get_velocity_features(user_id: str):
    query = """
        SELECT user_id, txn_count_10m, total_amount_10m,
               unique_merchants_10m, unique_countries_10m,
               window_start, window_end
        FROM velocity_features
        WHERE user_id = $1
        ORDER BY window_start DESC
        LIMIT 1
    """
    async with app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(query, user_id)

    if not row:
        return {"user_id": user_id, "found": False, "features": {}}

    return {
        "user_id": user_id,
        "found": True,
        "features": dict(row),
    }


@app.get("/features/historical/{user_id}")
async def get_historical_baseline(user_id: str):
    query = """
        SELECT
            user_id,
            AVG(daily_count) AS avg_daily_txn_count,
            STDDEV(daily_count) AS std_daily_txn_count,
            AVG(daily_amount) AS avg_daily_amount,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY daily_amount) AS p95_daily_amount
        FROM (
            SELECT
                user_id,
                DATE_TRUNC('day', timestamp) AS day,
                COUNT(*) AS daily_count,
                SUM(amount) AS daily_amount
            FROM transactions
            WHERE user_id = $1
              AND timestamp BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '1 hour'
            GROUP BY user_id, day
        ) daily_stats
        GROUP BY user_id
    """
    async with app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(query, user_id)

    if not row:
        return {"user_id": user_id, "found": False, "baseline": {}}

    return {
        "user_id": user_id,
        "found": True,
        "baseline": dict(row),
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
