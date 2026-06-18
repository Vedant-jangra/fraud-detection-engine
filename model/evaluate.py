import json
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

def compare_models(candidate_path: str, baseline_path: str,
                   test_data_path: str, output_path: str):
    candidate = joblib.load(candidate_path)
    baseline = joblib.load(baseline_path)
    
    if test_data_path.endswith('.parquet'):
        df = pd.read_parquet(test_data_path)
    else:
        df = pd.read_csv(test_data_path)
        
    from model.train import FEATURE_COLS, engineer_features
    
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)
    y = df['is_fraud']
    
    cand_auprc = average_precision_score(y, candidate.predict_proba(X)[:, 1])
    base_auprc = average_precision_score(y, baseline.predict_proba(X)[:, 1])
    
    result = {
        'candidate_auprc': float(cand_auprc),
        'baseline_auprc': float(base_auprc),
        'delta': float(cand_auprc - base_auprc),
        'promote': bool(cand_auprc >= base_auprc - 0.01)
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print("\nEvaluation Results:")
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == '__main__':
    import os
    os.makedirs('reports', exist_ok=True)
    
    compare_models(
        candidate_path='model/artifacts/xgb_model.pkl',
        baseline_path='model/artifacts/xgb_model.pkl',
        test_data_path='data/seeds/transactions_with_fraud_patterns.csv',
        output_path='reports/retrain_comparison.json'
    )
