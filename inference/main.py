from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncpg
import time
import logging
import os
import joblib
import shap
from prometheus_client import Counter, Histogram, make_asgi_app

from .schemas import TransactionRequest, FraudResponse, Explanation
from .predictor import FeatureExtractor, ONNXPredictor
from .explain import get_top_reasons

logger = logging.getLogger(__name__)

INFERENCE_LATENCY = Histogram(
    'inference_latency_ms', 'End-to-end inference latency in ms',
    buckets=[5, 10, 15, 20, 25, 30, 50, 100]
)
FRAUD_SCORE_HIST = Histogram(
    'fraud_score_distribution', 'Distribution of fraud probability scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
FRAUD_ALERTS = Counter('fraud_alerts_total', 'High-confidence fraud alerts')
REQUESTS_TOTAL = Counter('requests_total', 'Total inference requests')

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.environ.get('JDBC_URL', 'postgresql://postgres:fraud_engine_secret@localhost:5432/fraud_db')
    db_url = db_url.replace("jdbc:", "")
    
    try:
        app.state.db_pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=5, max_size=20,  
            command_timeout=5.0
        )
    except Exception as e:
        logger.warning(f"DB connection failed: {e}")
        app.state.db_pool = None
    
    onnx_path = os.environ.get('ONNX_MODEL_PATH', 'model/artifacts/model_quantized.onnx')
    xgb_path = os.environ.get('XGB_MODEL_PATH', 'model/artifacts/xgb_model.pkl')
    
    if not os.path.exists(onnx_path) and os.path.exists(xgb_path):
        import sys
        if '/app' not in sys.path:
            sys.path.append('/app')
        from model.export_onnx import export_to_onnx, quantize_model
        xgb_model = joblib.load(xgb_path)
        base_onnx = onnx_path.replace('_quantized', '')
        export_to_onnx(xgb_model, base_onnx)
        quantize_model(base_onnx, onnx_path)
    
    app.state.predictor = ONNXPredictor(onnx_path)
    if app.state.db_pool:
        app.state.extractor = FeatureExtractor(app.state.db_pool)
    else:
        app.state.extractor = None
    
    xgb_model = joblib.load(xgb_path)
    app.state.explainer = shap.TreeExplainer(xgb_model)
    
    yield
    
    if app.state.db_pool:
        await app.state.db_pool.close()

app = FastAPI(title='Fraud Intelligence Engine', version='1.0.0', lifespan=lifespan)
app.mount('/metrics', make_asgi_app())

@app.post('/v1/score', response_model=FraudResponse)
async def score_transaction(request: TransactionRequest):
    if not app.state.extractor:
        raise HTTPException(status_code=503, detail="Database connection failed.")
        
    t_start = time.perf_counter()
    REQUESTS_TOTAL.inc()
    
    features = await app.state.extractor.get_features(
        user_id=request.user_id,
        amount=request.amount,
        merchant_cat=request.merchant_cat,
        country=request.country,
        timestamp=request.timestamp,
    )
    
    fraud_prob = app.state.predictor.predict(features)
    latency_ms = (time.perf_counter() - t_start) * 1000
    
    INFERENCE_LATENCY.observe(latency_ms)
    FRAUD_SCORE_HIST.observe(fraud_prob)
    
    if fraud_prob > 0.7:
        FRAUD_ALERTS.inc()
        
    return FraudResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        risk_level='HIGH' if fraud_prob > 0.7 else 'MEDIUM' if fraud_prob > 0.3 else 'LOW',
        latency_ms=round(latency_ms, 2),
        features_used=features.feature_names,
    )

@app.post('/v1/explain', response_model=FraudResponse)
async def explain_transaction(request: TransactionRequest):
    if not app.state.extractor:
        raise HTTPException(status_code=503, detail="Database connection failed.")
        
    t_start = time.perf_counter()
    
    features = await app.state.extractor.get_features(
        user_id=request.user_id, amount=request.amount,
        merchant_cat=request.merchant_cat, country=request.country,
        timestamp=request.timestamp
    )
    
    fraud_prob = app.state.predictor.predict(features)
    
    shap_vals = app.state.explainer.shap_values(features.vector)[0]
    top_reasons = get_top_reasons(shap_vals, features.feature_names, n=3)
    
    latency_ms = (time.perf_counter() - t_start) * 1000
    
    explanation = Explanation(
        summary=f'Score driven primarily by {top_reasons[0]["feature"]}',
        top_factors=top_reasons,
        model_version='1.0.0',
        explainability_method='SHAP TreeExplainer'
    )
    
    return FraudResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        risk_level='HIGH' if fraud_prob > 0.7 else 'MEDIUM' if fraud_prob > 0.3 else 'LOW',
        latency_ms=round(latency_ms, 2),
        features_used=features.feature_names,
        explanation=explanation
    )

@app.get('/health')
async def health_check():
    return {'status': 'ok'}
