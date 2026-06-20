"""
IEEE-CIS Fraud Detection dataset loader and feature engineering.

Dataset: https://www.kaggle.com/c/ieee-fraud-detection/data
590,540 transactions | 433 features | 3.5% fraud rate

Place train_transaction.csv and train_identity.csv in data/ieee-cis/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_ieee_cis(data_dir: str = "data/ieee-cis") -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction + identity files."""
    data_path = Path(data_dir)
    txn_path = data_path / "train_transaction.csv"
    iden_path = data_path / "train_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"Transaction file not found at {txn_path}. "
            "Download from: kaggle.com/c/ieee-fraud-detection/data"
        )

    txn = pd.read_csv(txn_path)
    logger.info("Loaded transactions: %s", txn.shape)

    if iden_path.exists():
        iden = pd.read_csv(iden_path)
        logger.info("Loaded identity: %s", iden.shape)
        df = txn.merge(iden, on="TransactionID", how="left")
    else:
        logger.warning("Identity file not found, proceeding with transactions only")
        df = txn

    print(f"Shape: {df.shape} | Fraud rate: {df['isFraud'].mean():.3%}")
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast dtypes to reduce RAM from ~1.8GB to ~600MB."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(
        f"Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100 * (1 - end_mem / start_mem):.1f}% reduction)"
    )
    return df


def drop_high_null_cols(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Drop columns where > threshold fraction of values are missing."""
    null_rates = df.isnull().mean()
    drop_cols = null_rates[null_rates > threshold].index.tolist()
    print(f"Dropping {len(drop_cols)} columns with >{threshold:.0%} nulls")
    return df.drop(columns=drop_cols)


def check_leakage(df: pd.DataFrame, target_col: str = "isFraud") -> list[str]:
    """Flag features with suspiciously high correlation to target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    leaky = []
    for col in numeric_cols:
        corr = df[col].corr(df[target_col])
        if abs(corr) > 0.95:
            leaky.append((col, corr))
            print(f"  LEAKAGE WARNING: {col} corr={corr:.4f}")

    if not leaky:
        print("  No leakage detected (no feature with |corr| > 0.95)")
    return [col for col, _ in leaky]


def engineer_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency-encode card identifiers as fraud risk proxies."""
    df = df.copy()
    for col in ["card1", "card2", "card3", "card5"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).astype("float32")

    # Cross-feature: card1 + addr1 identifies unique cards more precisely
    if "card1" in df.columns and "addr1" in df.columns:
        df["uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        uid_freq = df["uid"].value_counts(normalize=True)
        df["uid_freq"] = df["uid"].map(uid_freq).astype("float32")

    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is relative seconds — extract cyclical features."""
    df = df.copy()
    if "TransactionDT" not in df.columns:
        return df

    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["day_of_week"] = ((df["TransactionDT"] // (3600 * 24)) % 7).astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype("int8")

    return df


def clean_device_info(df: pd.DataFrame) -> pd.DataFrame:
    """Parse DeviceInfo strings into usable features."""
    df = df.copy()
    if "DeviceInfo" not in df.columns:
        return df

    device = df["DeviceInfo"].fillna("unknown").str.lower()

    # Extract brand
    brands = ["samsung", "iphone", "huawei", "xiaomi", "lg", "motorola", "sony", "htc"]
    df["device_brand"] = "other"
    for brand in brands:
        df.loc[device.str.contains(brand, na=False), "device_brand"] = brand

    # Frequency encode the brand
    brand_freq = df["device_brand"].value_counts(normalize=True)
    df["device_brand_freq"] = df["device_brand"].map(brand_freq).astype("float32")

    return df


# The IEEE-CIS feature columns used for XGBoost training.
# V-columns are anonymized Vesta features — we keep low-null ones.
# Card/time/device features are engineered above.
IEEE_CIS_FEATURE_COLS = [
    # Amount
    "TransactionAmt",
    "log_amount",
    # Card frequency features
    "card1_freq",
    "card2_freq",
    "card3_freq",
    "card5_freq",
    "uid_freq",
    # Time features
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    # Card categorical (raw)
    "card4_enc",
    "card6_enc",
    # Product & email
    "ProductCD_enc",
    "P_emaildomain_enc",
    "R_emaildomain_enc",
    # Address
    "addr1",
    "addr2",
    # Distance
    "dist1",
    "dist2",
    # Count features
    "C1",
    "C2",
    "C4",
    "C5",
    "C6",
    "C13",
    "C14",
    # Device
    "device_brand_freq",
    # D (timedelta) features
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
    # V features (low-null, high-importance subset)
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V29",
    "V30",
    "V31",
    "V32",
    "V33",
    "V34",
    "V35",
    "V36",
    "V37",
    "V38",
    "V44",
    "V45",
    "V46",
    "V47",
    "V48",
    "V49",
    "V50",
    "V51",
    "V52",
    "V53",
    "V54",
    "V55",
    "V56",
    "V57",
    "V58",
    "V59",
    "V60",
    "V61",
    "V62",
    "V69",
    "V70",
    "V71",
    "V72",
    "V73",
    "V74",
    "V75",
    "V76",
    "V77",
    "V78",
    "V79",
    "V80",
    "V81",
    "V82",
    "V83",
    "V87",
    "V90",
    "V91",
    "V94",
    "V95",
    "V96",
    "V97",
    "V99",
    "V100",
    "V126",
    "V127",
    "V128",
    "V130",
    "V131",
    "V258",
    "V279",
    "V280",
    "V282",
    "V283",
    "V285",
    "V287",
    "V288",
    "V289",
    "V294",
    "V306",
    "V307",
    "V308",
    "V309",
    "V310",
    "V311",
    "V312",
    "V313",
    "V314",
    "V315",
    "V316",
    "V317",
    "V318",
    "V320",
    "V321",
]


def prepare_ieee_cis(data_dir: str = "data/ieee-cis") -> tuple[pd.DataFrame, list[str]]:
    """
    Full pipeline: load -> clean -> engineer -> return ready-to-train DataFrame.

    Returns:
        (df, feature_cols) — DataFrame with all features + list of column names to use.
    """
    print("=" * 60)
    print("IEEE-CIS Data Preparation Pipeline")
    print("=" * 60)

    # Load
    print("\n[1/7] Loading data...")
    df = load_ieee_cis(data_dir)

    # Memory optimization
    print("\n[2/7] Reducing memory...")
    df = reduce_memory(df)

    # Drop high-null columns
    print("\n[3/7] Dropping high-null columns...")
    df = drop_high_null_cols(df, threshold=0.9)

    # Leakage check
    print("\n[4/7] Checking for data leakage...")
    leaky_cols = check_leakage(df)
    if leaky_cols:
        df = df.drop(columns=leaky_cols)
        print(f"  Dropped {len(leaky_cols)} leaky columns")

    # Feature engineering
    print("\n[5/7] Engineering card features...")
    df = engineer_card_features(df)

    print("\n[6/7] Engineering time features...")
    df = engineer_time_features(df)
    df = clean_device_info(df)

    # Encode categoricals
    print("\n[7/7] Encoding categoricals...")
    df["log_amount"] = np.log1p(df["TransactionAmt"]).astype("float32")

    from sklearn.preprocessing import LabelEncoder

    for col in ["card4", "card6", "ProductCD", "P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))

    # Filter feature columns to only those actually present
    available_features = [c for c in IEEE_CIS_FEATURE_COLS if c in df.columns]
    missing = set(IEEE_CIS_FEATURE_COLS) - set(available_features)
    if missing:
        print(
            f"  Note: {len(missing)} expected features not in dataset (will use {len(available_features)})"
        )

    print(f"\nFinal shape: {df.shape}")
    print(f"Features: {len(available_features)}")
    print(f"Fraud rate: {df['isFraud'].mean():.3%}")
    print("=" * 60)

    return df, available_features


if __name__ == "__main__":
    df, features = prepare_ieee_cis()
    print("\nTop 10 features by non-null count:")
    for col in features[:10]:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} / {len(df):,} ({non_null / len(df):.1%})")
