import numpy as np
import pandas as pd
from prometheus_client import Gauge
import asyncpg
import logging

logger = logging.getLogger(__name__)

PSI_GAUGE = Gauge('feature_psi', 'Population Stability Index', ['feature_name'])
MODEL_PERF = Gauge('model_precision_rolling_1h', 'Rolling 1h precision')

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    _, bin_edges = np.histogram(expected, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bin_edges)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

async def run_drift_check(db_pool: asyncpg.Pool, training_stats: dict) -> dict:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            '''SELECT amount, txn_count_10m, velocity_ratio
               FROM transactions
               WHERE timestamp >= NOW() - INTERVAL '1 hour'
               LIMIT 10000'''
        )

    if not rows:
        logger.warning('No production data for drift check')
        return {}

    prod_df = pd.DataFrame(rows, columns=['amount', 'txn_count_10m', 'velocity_ratio'])

    drift_report = {}
    for feature in ['amount', 'txn_count_10m', 'velocity_ratio']:
        if feature not in training_stats:
            continue

        psi = compute_psi(
            expected=training_stats[feature],
            actual=prod_df[feature].dropna().values
        )
        PSI_GAUGE.labels(feature_name=feature).set(psi)
        drift_report[feature] = psi

        if psi > 0.2:
            logger.critical(
                f'DATA DRIFT ALERT: {feature} PSI={psi:.3f} > 0.2. '
                f'Model retraining required immediately!'
            )

    return drift_report

if __name__ == '__main__':
    ref = pd.read_csv('data/seeds/transactions.csv')
    cur = pd.read_csv('data/seeds/transactions_with_fraud_patterns.csv')

    for col in ['amount']:
        psi = compute_psi(ref[col].dropna().values, cur[col].dropna().values)
        label = 'STABLE' if psi < 0.1 else 'MODERATE DRIFT' if psi < 0.2 else 'MAJOR DRIFT'
        print(f'{col}: PSI={psi:.4f} ({label})')
