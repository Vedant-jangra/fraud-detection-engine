"""
XGBoost training pipeline for the IEEE-CIS Fraud Detection dataset.

Uses the same SMOTE + StratifiedKFold + AUPRC approach as the simulated
data pipeline, but with IEEE-CIS-specific feature engineering.

Usage:
    PYTHONPATH=. python model/train_ieee_cis.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


def train_ieee_cis_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "isFraud",
    output_path: str = "model/artifacts/xgb_ieee_cis.pkl",
) -> xgb.XGBClassifier:
    """
    Train XGBoost on IEEE-CIS data with SMOTE + 5-fold stratified CV.

    Returns the best fold model (last fold).
    """
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    fraud_count = int(y.sum())
    legit_count = len(y) - fraud_count
    pos_weight = legit_count / fraud_count if fraud_count > 0 else 1

    print(f"\nTraining data: {len(X):,} rows")
    print(f"Fraud: {fraud_count:,} ({fraud_count / len(X):.3%})")
    print(f"Legit: {legit_count:,}")
    print(f"scale_pos_weight: {pos_weight:.1f}")
    print(f"Features: {len(feature_cols)}")

    # SMOTE with lower k_neighbors for sparse fraud class
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": pos_weight,
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
    }

    models = []
    fold_scores = []

    print("\n" + "=" * 50)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        print(f"After SMOTE: {len(X_tr_sm):,} rows (from {len(X_tr):,})")

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr_sm, y_tr_sm, eval_set=[(X_val, y_val)], verbose=100)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)

        auprc = average_precision_score(y_val, oof_preds[val_idx])
        fold_scores.append(auprc)
        print(f"Fold {fold + 1} AUPRC: {auprc:.4f}")

    overall_auprc = average_precision_score(y, oof_preds)
    print("\n" + "=" * 50)
    print(f"Overall CV AUPRC: {overall_auprc:.4f}")
    print(f"Fold AUPRCs: {', '.join(f'{s:.4f}' for s in fold_scores)}")
    print(f"Std: {np.std(fold_scores):.4f}")

    # Save best model
    best_fold = int(np.argmax(fold_scores))
    best_model = models[best_fold]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"\nSaved best model (fold {best_fold + 1}) to {output_path}")

    return best_model


def print_feature_importance(
    model: xgb.XGBClassifier, feature_cols: list[str], top_n: int = 20
):
    """Print top-N features by gain importance."""
    importance = model.get_booster().get_score(importance_type="gain")
    # Map feature indices to names
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} Features by Gain:")
    print("-" * 45)
    for i, (fname, gain) in enumerate(sorted_feats[:top_n]):
        # XGBoost uses f0, f1, ... internally — map to column names
        if fname.startswith("f"):
            idx = int(fname[1:])
            if idx < len(feature_cols):
                fname = feature_cols[idx]
        print(f"  {i + 1:2d}. {fname:30s} {gain:.1f}")


if __name__ == "__main__":
    from data.loaders.ieee_cis_loader import prepare_ieee_cis

    # Prepare data
    df, feature_cols = prepare_ieee_cis()

    # Train model
    model = train_ieee_cis_xgboost(df, feature_cols)

    print("\n[SUCCESS] IEEE-CIS training complete!")
    print("Next: run 'python model/export_onnx.py' to generate ONNX artifacts")
