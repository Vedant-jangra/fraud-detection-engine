from pydantic import BaseModel
from typing import List, Optional

class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    timestamp: str
    amount: float
    merchant_cat: str
    country: str
    
class ExplanationFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str

class Explanation(BaseModel):
    summary: str
    top_factors: List[ExplanationFactor]
    model_version: str
    explainability_method: str

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    latency_ms: float
    features_used: List[str]
    explanation: Optional[Explanation] = None
