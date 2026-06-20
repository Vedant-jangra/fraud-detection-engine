import numpy as np
import pandas as pd

from data.loaders.ieee_cis_loader import (
    IEEE_CIS_FEATURE_COLS,
    check_leakage,
    clean_device_info,
    drop_high_null_cols,
    engineer_card_features,
    engineer_time_features,
    reduce_memory,
)


def _make_sample_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal DataFrame mimicking IEEE-CIS structure."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "isFraud": rng.choice([0, 1], size=n, p=[0.965, 0.035]),
            "TransactionDT": rng.integers(0, 86400 * 30, size=n),
            "TransactionAmt": rng.lognormal(4.5, 1.2, size=n).round(2),
            "card1": rng.integers(1000, 9999, size=n),
            "card2": rng.choice([100, 200, 300, np.nan], size=n),
            "card3": rng.choice([150, 185, np.nan], size=n),
            "card4": rng.choice(["visa", "mastercard", None], size=n),
            "card5": rng.choice([100, 200, 226, np.nan], size=n),
            "card6": rng.choice(["debit", "credit", None], size=n),
            "addr1": rng.choice([100, 200, 300, np.nan], size=n),
            "DeviceInfo": rng.choice(
                ["Samsung SM-G950U", "iPhone 8", None, "Huawei P30"],
                size=n,
            ),
            "V12": rng.random(n).astype("float64"),
            "V13": rng.random(n).astype("float64"),
            "high_null_col": [np.nan] * 95 + list(range(5)),
        }
    )


class TestReduceMemory:
    def test_reduces_float64_to_float32(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        assert df["a"].dtype == np.float64
        result = reduce_memory(df)
        assert result["a"].dtype == np.float32

    def test_downcasts_int64(self):
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype="int64")})
        result = reduce_memory(df)
        assert result["a"].dtype in (np.int8, np.int16, np.int32)


class TestDropHighNullCols:
    def test_drops_columns_above_threshold(self):
        df = _make_sample_df()
        assert "high_null_col" in df.columns
        result = drop_high_null_cols(df, threshold=0.9)
        assert "high_null_col" not in result.columns

    def test_keeps_columns_below_threshold(self):
        df = _make_sample_df()
        result = drop_high_null_cols(df, threshold=0.9)
        assert "TransactionAmt" in result.columns
        assert "card1" in result.columns


class TestCheckLeakage:
    def test_detects_perfectly_correlated_feature(self):
        df = pd.DataFrame({"isFraud": [0, 1, 0, 1, 0], "leak": [0, 1, 0, 1, 0]})
        leaky = check_leakage(df)
        assert "leak" in leaky

    def test_no_leakage_on_random_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "isFraud": rng.choice([0, 1], size=100),
                "random_feat": rng.random(100),
            }
        )
        leaky = check_leakage(df)
        assert len(leaky) == 0


class TestEngineerCardFeatures:
    def test_creates_frequency_columns(self):
        df = _make_sample_df()
        result = engineer_card_features(df)
        assert "card1_freq" in result.columns
        assert "uid_freq" in result.columns
        assert result["card1_freq"].dtype == np.float32

    def test_uid_combines_card1_and_addr1(self):
        df = _make_sample_df()
        result = engineer_card_features(df)
        assert "uid" in result.columns


class TestEngineerTimeFeatures:
    def test_extracts_hour_and_day(self):
        df = _make_sample_df()
        result = engineer_time_features(df)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "is_night" in result.columns
        assert result["hour"].between(0, 23).all()
        assert result["day_of_week"].between(0, 6).all()

    def test_returns_unchanged_if_no_transaction_dt(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = engineer_time_features(df)
        assert "hour" not in result.columns


class TestCleanDeviceInfo:
    def test_extracts_known_brands(self):
        df = _make_sample_df()
        result = clean_device_info(df)
        assert "device_brand" in result.columns
        assert "device_brand_freq" in result.columns
        brands = result["device_brand"].unique()
        assert "samsung" in brands or "iphone" in brands

    def test_labels_unknown_as_other(self):
        df = pd.DataFrame({"DeviceInfo": [None, None]})
        result = clean_device_info(df)
        assert (result["device_brand"] == "other").all()


class TestFeatureColsList:
    def test_feature_cols_list_is_nonempty(self):
        assert len(IEEE_CIS_FEATURE_COLS) > 50

    def test_feature_cols_contain_v_columns(self):
        v_cols = [c for c in IEEE_CIS_FEATURE_COLS if c.startswith("V")]
        assert len(v_cols) > 30

    def test_feature_cols_contain_engineered_features(self):
        assert "log_amount" in IEEE_CIS_FEATURE_COLS
        assert "hour" in IEEE_CIS_FEATURE_COLS
        assert "card1_freq" in IEEE_CIS_FEATURE_COLS
