# api/app.py

# --- PART 1: IMPORTS AND SETUP ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os

# Initialize the FastAPI app
app = FastAPI(
    title="FraudLens API",
    description="Real-time fraud detection with explainability",
    version="1.0.0"
)

# --- PART 2: DEFINE THE INPUT FORMAT ---
# This tells the API exactly what a transaction looks like

class Transaction(BaseModel):
    TransactionAmt: float
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    C6: float
    C7: float
    C8: float
    C9: float
    C10: float
    C11: float
    C12: float
    C13: float
    C14: float
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D6: Optional[float] = None
    D7: Optional[float] = None
    D8: Optional[float] = None
    D9: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D15: Optional[float] = None
    TransactionDT: int
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    id_31: Optional[str] = None  # browser

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 500,
                "card1": 13926,
                "card2": 404,
                "card3": 150,
                "card5": 142,
                "addr1": 315,
                "addr2": 87,
                "dist1": 500,
                "C1": 10, "C2": 8, "C3": 0, "C4": 0,
                "C5": 0, "C6": 5, "C7": 0, "C8": 3,
                "C9": 5, "C10": 0, "C11": 5, "C12": 0,
                "C13": 15, "C14": 8,
                "D1": 0,
                "TransactionDT": 25200,
                "ProductCD": "C",
                "card4": "discover",
                "card6": "credit",
                "P_emaildomain": "outlook.com",
                "DeviceType": "mobile",
                "id_31": "chrome 65.0"
            }
        }

# --- PART 3: LOAD MODEL ON STARTUP ---

# Paths to saved files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_v5.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_cols_v5.json")

# Load model and feature list
model = joblib.load(MODEL_PATH)
with open(FEATURE_PATH, "r") as f:
    feature_cols = json.load(f)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# ISP email list (from our feature engineering)
ISP_DOMAINS = [
    'sbcglobal.net', 'att.net', 'verizon.net', 'cox.net',
    'comcast.net', 'bellsouth.net', 'optonline.net', 'charter.net',
    'frontier.com', 'frontiernet.net', 'centurylink.net'
]

# Store predictions for monitoring
prediction_log = []

print(f"Model loaded with {len(feature_cols)} features")

# --- PART 4: FEATURE ENGINEERING FUNCTION ---
# Same transformations we did in the notebook, but as a reusable function

def engineer_features(txn: Transaction) -> pd.DataFrame:
    """Convert raw transaction into the feature vector the model expects."""
    
    data = txn.model_dump()
    
    # Time features
    data['hour'] = (data['TransactionDT'] / 3600) % 24
    data['weekday'] = int((data['TransactionDT'] / 86400) % 7)
    data['is_night'] = int(data['hour'] >= 0 and data['hour'] <= 9)
    
    # Amount features
    data['TransactionAmt_log'] = np.log1p(data['TransactionAmt'])
    data['TransactionAmt_decimal'] = data['TransactionAmt'] - int(data['TransactionAmt'])
    data['is_round_amount'] = int(data['TransactionAmt_decimal'] == 0)
    
    # Email features
    data['P_email_is_isp'] = int(data.get('P_emaildomain') in ISP_DOMAINS)
    data['R_email_is_isp'] = int(data.get('R_emaildomain') in ISP_DOMAINS)
    data['email_domain_match'] = int(
        data.get('P_emaildomain') is not None and 
        data.get('R_emaildomain') is not None and
        data['P_emaildomain'] == data['R_emaildomain']
    )
    data['P_email_missing'] = int(data.get('P_emaildomain') is None)
    data['R_email_missing'] = int(data.get('R_emaildomain') is None)
    
    # Identity features
    data['has_identity'] = int(data.get('DeviceType') is not None)
    data['is_mobile'] = int(data.get('DeviceType') == 'mobile')
    
    # Browser cleaning
    browser = str(data.get('id_31', 'unknown')).lower()
    if 'chrome' in browser: browser_clean = 'chrome'
    elif 'safari' in browser: browser_clean = 'safari'
    elif 'firefox' in browser: browser_clean = 'firefox'
    elif 'edge' in browser: browser_clean = 'edge'
    elif 'ie' in browser or 'explorer' in browser: browser_clean = 'ie'
    elif 'samsung' in browser: browser_clean = 'samsung'
    else: browser_clean = 'other'
    
    # Label encode categoricals (simple mapping)
    # In production you'd save the encoders, for now use hash-based encoding
    cat_mappings = {
        'ProductCD_encoded': data.get('ProductCD', 'missing'),
        'card4_encoded': data.get('card4', 'missing'),
        'card6_encoded': data.get('card6', 'missing'),
        'P_emaildomain_encoded': data.get('P_emaildomain', 'missing'),
        'R_emaildomain_encoded': data.get('R_emaildomain', 'missing'),
        'DeviceType_encoded': data.get('DeviceType', 'missing'),
        'browser_clean': browser_clean,
    }
    for key, val in cat_mappings.items():
        data[key] = hash(str(val)) % 10000  # simple hash encoding
    
    # Card-level stats (in production, these come from a feature store)
    # For now, use placeholder values — we'll improve this later
    data['card1_amt_mean'] = data['TransactionAmt']  # placeholder
    data['card1_amt_std'] = 0.0
    data['card1_txn_count'] = 1
    data['card1_amt_deviation'] = 0.0
    
    # Address amount stats (placeholder)
    data['addr1_amt_mean'] = data['TransactionAmt']
    data['addr1_amt_std'] = 0.0
    
    # Derived features
    data['card1_txn_per_day'] = 0.0
    data['amt_x_is_night'] = data['TransactionAmt_log'] * data['is_night']
    data['amt_x_is_mobile'] = data['TransactionAmt_log'] * data['is_mobile']
    data['C1_C2_ratio'] = data['C1'] / (data['C2'] + 1e-6)
    data['C13_C14_ratio'] = data['C13'] / (data['C14'] + 1e-6)
    data['D1_missing'] = int(data.get('D1') is None)
    data['M_true_count'] = 0  # placeholder — no M columns in input
    
    # V features and v_block features — set to NaN (missing)
    # The model handles NaN natively via LightGBM
    for col in feature_cols:
        if col not in data:
            data[col] = np.nan
    
    # Build DataFrame with correct column order
    df = pd.DataFrame([{col: data.get(col, np.nan) for col in feature_cols}])
    
    # Ensure all columns are numeric (fixes None → NaN conversion)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df

# --- PART 5: API ENDPOINTS ---

@app.get("/health")
def health_check():
    """Check if the API is running and model is loaded."""
    return {
        "status": "healthy",
        "model_features": len(feature_cols),
        "predictions_served": len(prediction_log)
    }


@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict fraud probability for a transaction.
    Returns probability, risk level, and SHAP explanation.
    """
    try:
        # Engineer features
        features_df = engineer_features(transaction)
        
        # Predict
        fraud_prob = model.predict_proba(features_df)[0][1]
        
        # SHAP explanation
        shap_values = explainer.shap_values(features_df)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        
        # Get top contributing features
        feature_importance = sorted(
            zip(feature_cols, sv.tolist(), features_df.iloc[0].tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        reasons = []
        for feat, shap_val, feat_val in feature_importance:
            reasons.append({
                "feature": feat,
                "value": round(float(feat_val), 4) if not np.isnan(feat_val) else None,
                "impact": round(float(shap_val), 4),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk"
            })
        
        # Risk level
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Log prediction for monitoring
        prediction_log.append({
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "amount": transaction.TransactionAmt
        })
        
        return {
            "fraud_probability": round(float(fraud_prob), 4),
            "risk_level": risk_level,
            "reasons": reasons,
            "transaction_amount": transaction.TransactionAmt
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Get summary statistics of predictions served so far."""
    if not prediction_log:
        return {"message": "No predictions yet"}
    
    probs = [p["fraud_probability"] for p in prediction_log]
    amounts = [p["amount"] for p in prediction_log]
    risk_counts = {}
    for p in prediction_log:
        risk_counts[p["risk_level"]] = risk_counts.get(p["risk_level"], 0) + 1
    
    return {
        "total_predictions": len(prediction_log),
        "avg_fraud_probability": round(float(np.mean(probs)), 4),
        "max_fraud_probability": round(float(np.max(probs)), 4),
        "avg_transaction_amount": round(float(np.mean(amounts)), 2),
        "risk_distribution": risk_counts
    }

