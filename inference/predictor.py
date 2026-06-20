import numpy as np
import onnxruntime as rt
import asyncpg

from sklearn.preprocessing import LabelEncoder
import pandas as pd


class FeatureVector:
    def __init__(self, vector: np.ndarray, feature_names: list):
        self.vector = vector
        self.feature_names = feature_names


class ONNXPredictor:
    def __init__(self, model_path: str):
        self.session = rt.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: FeatureVector) -> float:
        result = self.session.run(
            None, {self.input_name: features.vector.astype(np.float32)}
        )
        probabilities = result[1][0]
        # ONNX returns a numpy array of [prob_legit, prob_fraud]
        return float(probabilities[1])


class FeatureExtractor:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.le = LabelEncoder()
        self.le.fit(
            [
                "grocery",
                "travel",
                "electronics",
                "restaurant",
                "utility",
                "crypto",
                "unknown",
                "XX",
                "IN",
                "US",
                "SG",
                "AE",
            ]
        )

        self.feature_names = [
            "log_amount",
            "amount_rounded",
            "is_micro_txn",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_night",
            "merchant_cat_enc",
            "country_enc",
            "txn_count_10m",
            "total_amount_10m",
            "velocity_ratio",
            "amount_spike_ratio",
            "unique_merchants_10m",
            "unique_countries_10m",
            "txn_count_1h",
        ]

    def _encode_categorical(self, value: str, fallback: str) -> int:
        try:
            return int(self.le.transform([value])[0])
        except ValueError:
            return int(self.le.transform([fallback])[0])

    async def get_features(
        self,
        user_id: str,
        amount: float,
        merchant_cat: str,
        country: str,
        timestamp: str,
    ) -> FeatureVector:
        dt = pd.to_datetime(timestamp)

        log_amount = np.log1p(amount)
        amount_rounded = 1 if amount % 1 == 0 else 0
        is_micro_txn = 1 if amount < 10 else 0

        hour_of_day = dt.hour
        day_of_week = dt.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if 22 <= hour_of_day or hour_of_day <= 6 else 0

        merchant_cat_enc = self._encode_categorical(merchant_cat, "unknown")
        country_enc = self._encode_categorical(country, "XX")

        query = """
            SELECT txn_count_10m, total_amount_10m, unique_merchants_10m, unique_countries_10m
            FROM velocity_features
            WHERE user_id = $1
            ORDER BY window_start DESC
            LIMIT 1
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id)

        if row:
            txn_count_10m = row["txn_count_10m"]
            total_amount_10m = row["total_amount_10m"]
            unique_merchants_10m = row["unique_merchants_10m"]
            unique_countries_10m = row["unique_countries_10m"]
        else:
            txn_count_10m = 0
            total_amount_10m = 0.0
            unique_merchants_10m = 0
            unique_countries_10m = 0

        velocity_ratio = 1.0
        amount_spike_ratio = 1.0
        txn_count_1h = txn_count_10m

        vector = np.array(
            [
                [
                    log_amount,
                    amount_rounded,
                    is_micro_txn,
                    hour_of_day,
                    day_of_week,
                    is_weekend,
                    is_night,
                    merchant_cat_enc,
                    country_enc,
                    txn_count_10m,
                    total_amount_10m,
                    velocity_ratio,
                    amount_spike_ratio,
                    unique_merchants_10m,
                    unique_countries_10m,
                    txn_count_1h,
                ]
            ]
        )

        return FeatureVector(vector, self.feature_names)
