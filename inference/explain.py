import numpy as np


def get_top_reasons(
    shap_values_row: np.ndarray, feature_names: list, n: int = 3
) -> list[dict]:
    pairs = list(zip(feature_names, shap_values_row))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature": name,
            "shap_value": round(float(val), 4),
            "direction": "increases_fraud_risk" if val > 0 else "decreases_fraud_risk",
        }
        for name, val in pairs[:n]
    ]
