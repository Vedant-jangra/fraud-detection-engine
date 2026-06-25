import numpy as np


def get_top_reasons(
    shap_values_row: np.ndarray, feature_names: list, n: int = 3
) -> list[dict]:
    """Return top-N features driving a single fraud prediction by SHAP impact.

    Used by the /v1/explain API endpoint to provide human-readable
    feature attributions for compliance with RBI guidelines.

    Args:
        shap_values_row: SHAP values array for one prediction.
        feature_names: List of feature names matching the SHAP array.
        n: Number of top factors to return.

    Returns:
        List of dicts with feature name, SHAP value, and direction.
    """
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
