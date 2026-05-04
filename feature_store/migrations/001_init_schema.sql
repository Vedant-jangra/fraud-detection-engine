-- 001_init_schema.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- transactions table
-- PRIMARY KEY is (transaction_id, timestamp) because TimescaleDB
-- requires the partitioning column in the primary key
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  UUID           NOT NULL DEFAULT gen_random_uuid(),
    user_id         TEXT           NOT NULL,
    timestamp       TIMESTAMPTZ    NOT NULL,
    amount          NUMERIC(12, 2) NOT NULL,
    merchant_id     TEXT,
    merchant_cat    TEXT,
    country         CHAR(2),
    device_id       TEXT,
    ip_address      INET,
    is_fraud        SMALLINT       DEFAULT 0,
    PRIMARY KEY (transaction_id, timestamp)
);

SELECT create_hypertable(
    'transactions',
    'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_txn_user_time
    ON transactions (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_txn_device
    ON transactions (device_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_txn_ip
    ON transactions (ip_address, timestamp DESC);

-- velocity features table
CREATE TABLE IF NOT EXISTS velocity_features (
    user_id                 TEXT           NOT NULL,
    window_start            TIMESTAMPTZ    NOT NULL,
    window_end              TIMESTAMPTZ    NOT NULL,
    txn_count_10m           INTEGER        DEFAULT 0,
    total_amount_10m        NUMERIC(12, 2) DEFAULT 0,
    unique_merchants_10m    INTEGER        DEFAULT 0,
    unique_countries_10m    INTEGER        DEFAULT 0,
    created_at              TIMESTAMPTZ    DEFAULT NOW(),
    PRIMARY KEY (user_id, window_start)
);

SELECT create_hypertable(
    'velocity_features',
    'window_start',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- continuous aggregate on transactions hypertable
CREATE MATERIALIZED VIEW IF NOT EXISTS user_hourly_stats
WITH (timescaledb.continuous) AS
    SELECT
        user_id,
        time_bucket('1 hour', timestamp) AS bucket,
        COUNT(*)                          AS txn_count,
        SUM(amount)                       AS total_amount,
        AVG(amount)                       AS avg_amount,
        MAX(amount)                       AS max_amount,
        COUNT(*) FILTER (WHERE is_fraud = 1) AS fraud_count
    FROM transactions
    GROUP BY user_id, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'user_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '30 seconds',
    schedule_interval => INTERVAL '30 seconds'
);

SELECT hypertable_name, num_chunks
FROM timescaledb_information.hypertables;
