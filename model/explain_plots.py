import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_shap_plots():
    print("Loading model and training data...")
    model = joblib.load("model/artifacts/xgb_model.pkl")
    
    from data.generators.transaction_simulator import generate_dataset
    from sklearn.preprocessing import LabelEncoder
    
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
            "txn_count_10m", "total_amount_10m", "velocity_ratio", "amount_spike_ratio",
            "unique_merchants_10m", "unique_countries_10m", "txn_count_1h",
        ]
        for col in velocity_cols:
            df[col] = df.get(col, pd.Series(np.zeros(len(df))))
        return df
    
    # We need a sample to generate SHAP plots
    print("Generating sample data for SHAP beeswarm plot...")
    df_raw = generate_dataset(n_transactions=1000, fraud_rate=0.05)
    
    df_features = engineer_features(df_raw)
    
    feature_cols = [
        "log_amount", "amount_rounded", "is_micro_txn", "hour_of_day", "day_of_week",
        "is_weekend", "is_night", "merchant_cat_enc", "country_enc",
        "txn_count_10m", "total_amount_10m", "velocity_ratio", "amount_spike_ratio",
        "unique_merchants_10m", "unique_countries_10m", "txn_count_1h"
    ]
    
    X = df_features[feature_cols]
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print("Plotting SHAP summary...")
    os.makedirs("reports", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance (Fraud Engine)", fontsize=16)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Saved SHAP summary to reports/shap_summary.png")

if __name__ == "__main__":
    generate_shap_plots()
