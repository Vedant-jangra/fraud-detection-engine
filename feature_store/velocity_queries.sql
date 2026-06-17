-- velocity_queries.sql
-- Reusable investigation queries for high-velocity fraud patterns.

-- 1) Card testing detector:
-- Flags users making many small payments in a rolling 10-minute window.
WITH small_payments AS (
    SELECT
        transaction_id,
        user_id,
        timestamp,
        amount,
        merchant_id,
        merchant_cat,
        device_id,
        ip_address,
        COUNT(*) OVER (
            PARTITION BY user_id
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL '10 minutes' PRECEDING AND CURRENT ROW
        ) AS txn_count_10m,
        SUM(amount) OVER (
            PARTITION BY user_id
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL '10 minutes' PRECEDING AND CURRENT ROW
        ) AS total_amount_10m
    FROM transactions
    WHERE amount BETWEEN 1 AND 50
)
SELECT
    transaction_id,
    user_id,
    timestamp,
    amount,
    merchant_id,
    merchant_cat,
    device_id,
    ip_address,
    txn_count_10m,
    total_amount_10m,
    'card_testing' AS suspected_pattern
FROM small_payments
WHERE txn_count_10m >= 20
ORDER BY timestamp DESC;

-- 2) Account takeover detector:
-- Flags users with high-value transactions from multiple countries/devices in one hour.
WITH hourly_user_activity AS (
    SELECT
        user_id,
        time_bucket('1 hour', timestamp) AS bucket,
        COUNT(*) AS txn_count_1h,
        SUM(amount) AS total_amount_1h,
        MAX(amount) AS max_amount_1h,
        COUNT(DISTINCT country) AS country_count_1h,
        COUNT(DISTINCT device_id) AS device_count_1h,
        COUNT(DISTINCT ip_address) AS ip_count_1h
    FROM transactions
    GROUP BY user_id, bucket
)
SELECT
    user_id,
    bucket,
    txn_count_1h,
    total_amount_1h,
    max_amount_1h,
    country_count_1h,
    device_count_1h,
    ip_count_1h,
    'account_takeover' AS suspected_pattern
FROM hourly_user_activity
WHERE max_amount_1h >= 5000
  AND (country_count_1h > 1 OR device_count_1h > 1 OR ip_count_1h > 2)
ORDER BY bucket DESC;

-- 3) Bust-out detector:
-- Flags users whose latest hour is much larger than their historical hourly average.
WITH hourly_amounts AS (
    SELECT
        user_id,
        time_bucket('1 hour', timestamp) AS bucket,
        SUM(amount) AS total_amount_1h,
        COUNT(*) AS txn_count_1h
    FROM transactions
    GROUP BY user_id, bucket
),
hourly_baseline AS (
    SELECT
        user_id,
        bucket,
        total_amount_1h,
        txn_count_1h,
        AVG(total_amount_1h) OVER (
            PARTITION BY user_id
            ORDER BY bucket
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS avg_previous_hourly_amount
    FROM hourly_amounts
)
SELECT
    user_id,
    bucket,
    txn_count_1h,
    total_amount_1h,
    avg_previous_hourly_amount,
    total_amount_1h / NULLIF(avg_previous_hourly_amount, 0) AS spike_ratio,
    'bust_out' AS suspected_pattern
FROM hourly_baseline
WHERE avg_previous_hourly_amount IS NOT NULL
  AND total_amount_1h >= 10000
  AND total_amount_1h >= avg_previous_hourly_amount * 5
ORDER BY bucket DESC;

-- 4) Spark velocity feature inspection:
-- Reads features produced by processing/spark_streaming.py.
SELECT
    user_id,
    window_start,
    window_end,
    txn_count_10m,
    total_amount_10m,
    unique_merchants_10m,
    unique_countries_10m
FROM velocity_features
WHERE txn_count_10m >= 20
   OR total_amount_10m >= 50000
   OR unique_countries_10m > 1
ORDER BY window_start DESC;
