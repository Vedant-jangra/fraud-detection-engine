import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer temporal, amount, and velocity features from raw transaction data.

    Args:
        df: Raw transaction DataFrame with columns: timestamp, amount,
            merchant_cat, country, and optionally pre-joined velocity columns.

    Returns:
        DataFrame with engineered feature columns appended.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = df["hour_of_day"].between(22, 6).astype(int)

    df["log_amount"] = np.log1p(df["amount"])
    df["amount_rounded"] = (df["amount"] % 1 == 0).astype(int)
    df["is_micro_txn"] = (df["amount"] < 10).astype(int)

    le = LabelEncoder()
    df["merchant_cat_enc"] = le.fit_transform(df["merchant_cat"].fillna("unknown"))
    df["country_enc"] = le.fit_transform(df["country"].fillna("XX"))

    velocity_cols = [
        "txn_count_10m",
        "total_amount_10m",
        "velocity_ratio",
        "amount_spike_ratio",
        "unique_merchants_10m",
        "unique_countries_10m",
        "txn_count_1h",
    ]
    for col in velocity_cols:
        df[col] = df.get(col, pd.Series(np.zeros(len(df))))

    return df


FEATURE_COLS = [
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


def train_xgboost(df: pd.DataFrame):
    """Train an XGBoost classifier with SMOTE oversampling and 5-fold stratified CV.

    Handles class imbalance via scale_pos_weight and SMOTE on training folds.
    Optimizes for AUPRC (Area Under Precision-Recall Curve), the correct metric
    for severely imbalanced fraud detection.

    Args:
        df: Transaction DataFrame with 'is_fraud' label column.

    Returns:
        Trained XGBClassifier from the last fold.
    """
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]

    fraud_count = y.sum()
    legit_count = len(y) - fraud_count
    pos_weight = legit_count / fraud_count if fraud_count > 0 else 1

    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": pos_weight,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "early_stopping_rounds": 30,
        "random_state": 42,
    }

    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr_sm, y_tr_sm, eval_set=[(X_val, y_val)], verbose=50)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)

        auprc = average_precision_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold+1} AUPRC: {auprc:.4f}")

    overall_auprc = average_precision_score(y, oof_preds)
    print(f"Overall CV AUPRC: {overall_auprc:.4f}")

    return models[-1]


if __name__ == "__main__":
    dataset_path = "data/seeds/transactions_with_fraud_patterns.csv"
    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    model = train_xgboost(df)

    os.makedirs("model/artifacts", exist_ok=True)
    artifact_path = "model/artifacts/xgb_model.pkl"
    joblib.dump(model, artifact_path)
