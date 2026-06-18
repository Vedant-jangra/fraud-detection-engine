-- velocity_queries.sql
-- Unified parameterized velocity feature query with 4 CTEs.

WITH

recent_activity AS (
    SELECT
        user_id,
        COUNT(*)                          AS txn_count_10m,
        SUM(amount)                       AS total_amount_10m,
        AVG(amount)                       AS avg_amount_10m,
        MAX(amount)                       AS max_amount_10m,
        COUNT(DISTINCT merchant_id)       AS unique_merchants_10m,
        COUNT(DISTINCT country)           AS unique_countries_10m,
        COUNT(DISTINCT device_id)         AS unique_devices_10m
    FROM transactions
    WHERE
        user_id = $1
        AND timestamp >= NOW() - INTERVAL '10 minutes'
    GROUP BY user_id
),

historical_baseline AS (
    SELECT
        user_id,
        AVG(daily_count)                  AS avg_daily_txn_count,
        STDDEV(daily_count)               AS std_daily_txn_count,
        AVG(daily_amount)                 AS avg_daily_amount,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY daily_amount)
                                          AS p95_daily_amount
    FROM (
        SELECT
            user_id,
            DATE_TRUNC('day', timestamp)  AS day,
            COUNT(*)                      AS daily_count,
            SUM(amount)                   AS daily_amount
        FROM transactions
        WHERE
            user_id = $1
            AND timestamp BETWEEN NOW() - INTERVAL '30 days'
                                AND NOW() - INTERVAL '1 hour'
        GROUP BY user_id, day
    ) daily_stats
    GROUP BY user_id
),

window_velocity AS (
    SELECT
        user_id, timestamp, amount,
        COUNT(*) OVER (
            PARTITION BY user_id ORDER BY timestamp
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        )                                    AS txn_count_1h,
        COUNT(*) OVER (
            PARTITION BY user_id ORDER BY timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        )                                    AS txn_count_24h,
        (amount - AVG(amount) OVER (PARTITION BY user_id)) /
            NULLIF(STDDEV(amount) OVER (PARTITION BY user_id), 0)
                                             AS amount_zscore,
        timestamp - LAG(timestamp) OVER (
            PARTITION BY user_id ORDER BY timestamp
        )                                    AS time_since_last_txn
    FROM transactions
    WHERE user_id = $1 AND timestamp >= NOW() - INTERVAL '25 hours'
),

velocity_ratio AS (
    SELECT
        r.user_id,
        r.txn_count_10m,  r.total_amount_10m,
        r.unique_merchants_10m, r.unique_countries_10m,
        h.avg_daily_txn_count,  h.avg_daily_amount,
        CASE WHEN COALESCE(h.avg_daily_txn_count, 0) = 0 THEN NULL
             ELSE (r.txn_count_10m * 144.0) / h.avg_daily_txn_count
        END                                  AS velocity_ratio,
        CASE WHEN COALESCE(h.avg_daily_amount, 0) = 0 THEN NULL
             ELSE r.total_amount_10m / h.avg_daily_amount
        END                                  AS amount_spike_ratio
    FROM recent_activity r
    LEFT JOIN historical_baseline h USING (user_id)
)

SELECT
    vr.*,
    (velocity_ratio > 10)::int             AS high_velocity_flag,
    (amount_spike_ratio > 5)::int          AS amount_spike_flag
FROM velocity_ratio vr;
