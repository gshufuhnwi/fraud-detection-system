from pathlib import Path
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import requests
import os
import joblib
import pandas as pd
import shap
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import numpy as np
from app.bank import BankSystem
from app.models import (
    CreateAccountRequest,
    DepositWithdrawRequest,
    FraudTransactionRequest,
    TransferRequest,
)

app = FastAPI(title="Banking Fraud System API")

bank = BankSystem()


ALERT_EMAIL = os.getenv("ALERT_EMAIL", "your_email@gmail.com")
ALERT_PASSWORD = os.getenv("ALERT_PASSWORD")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
LAST_ALERT_TIME = 0
ALERT_COOLDOWN = 30
if not ALERT_PASSWORD:
    print("WARNING: ALERT_PASSWORD not set")
# ✅ ADD HELPER HERE
def clean_for_json(data):
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return data
    else:
        return data

def send_email_alert(message: str):
    try:
        msg = MIMEText(message)
        msg["Subject"] = "🚨 Fraud Alert"
        msg["From"] = ALERT_EMAIL
        msg["To"] = ALERT_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(ALERT_EMAIL, ALERT_PASSWORD)
            server.send_message(msg)

    except Exception as e:
        print("EMAIL ALERT FAILED:", e)

def send_slack_alert(message: str):
    try:
        if not SLACK_WEBHOOK_URL:
            return

        payload = {"text": f"🚨 *Fraud Alert*\n{message}"}
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)

    except Exception as e:
        print("SLACK ALERT FAILED:", e)
# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
SCORED_FILE = DATA_DIR / "scored_transactions.csv"
LOG_FILE = LOG_DIR / "predictions.jsonl"

# =========================================================
# LOAD MODEL BUNDLE
# Expected keys:
# - pipeline
# - feature_columns
# - transformed_feature_names
# - threshold
# =========================================================
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
transformed_feature_names = bundle["transformed_feature_names"]
threshold = bundle.get("threshold", 0.5)

preprocessor = pipeline.named_steps["preprocessor"]
tree_model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(tree_model)


# =========================================================
# HELPERS
# =========================================================
def decide_action(probability: float) -> str:
    if probability >= 0.85:
        return "block"
    elif probability >= 0.60:
        return "manual_review"
    elif probability >= 0.40:
        return "step_up_auth"
    return "approve"
def log_prediction(payload: dict, prediction: int, probability: float) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **payload,
        "prediction": int(prediction),
        "probability": float(probability),
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def score_transaction(payload: dict) -> dict:
    try:
        input_df = pd.DataFrame([payload])

        missing = [c for c in feature_columns if c not in input_df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        input_df = input_df[feature_columns]

        probability = float(pipeline.predict_proba(input_df)[0, 1])
        prediction = int(pipeline.predict(input_df)[0])

        risk_level = (
            "high" if probability >= 0.80
            else "medium" if probability >= threshold
            else "low"
        )

        # SHAP on transformed features
        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        shap_raw = explainer.shap_values(X_transformed)

        if isinstance(shap_raw, list):
            row_shap = shap_raw[1][0]
        else:
            shape = getattr(shap_raw, "shape", None)
            if shape is not None and len(shape) == 3:
                row_shap = shap_raw[0, :, 1]
            else:
                row_shap = shap_raw[0]

        shap_top_features = []
        for feature_name, shap_value in zip(transformed_feature_names, row_shap):
            shap_top_features.append(
                {
                    "feature": str(feature_name),
                    "shap_value": float(shap_value),
                }
            )

        shap_top_features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        shap_top_features = shap_top_features[:10]

        log_prediction(payload, prediction, probability)
        action= decide_action(probability)
        customer_id = payload.get("customer_id", "unknown")
        account_id = payload.get("account_id", "unknown")
        if probability>=0.90:
            severity = "HIGH"
        elif probability>=0.75:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        global LAST_ALERT_TIME
        current_time = time.time()
        
        if probability >= 0.80 and (current_time - LAST_ALERT_TIME > ALERT_COOLDOWN):
            alert_msg = f"""
            Customer: {customer_id}
            Account: {account_id}
            Fraud Probability: {probability:.2f}
            Prediction: {'FRAUD' if prediction == 1 else 'LEGIT'}
            Action: {action}
            Severity: {severity}
            """
            if ALERT_EMAIL and ALERT_PASSWORD:
                try:
                    send_email_alert(alert_msg)
                except Exception as e:
                    print("Email alert error:", e)
            if SLACK_WEBHOOK_URL:
                try:
                    send_slack_alert(alert_msg)
                except Exception as e:
                    print("Slack alert error:", e)
            LAST_ALERT_TIME = current_time

        result = {
            "customer_id": customer_id,
            "account_id": account_id,
            "fraud_probability": round(probability, 6),
            "fraud_prediction": prediction,
            "predicted_label": "fraud" if prediction == 1 else "legitimate",
            "risk_level": risk_level,
            "recommended_action": action,
            "shap_top_features": shap_top_features,
        }
        log_record = {
    "timestamp": datetime.utcnow().isoformat(),
    "customer_id": customer_id,
    "account_id": account_id,
    "fraud_probability": result["fraud_probability"],
    "fraud_prediction": result["fraud_prediction"],
    "predicted_label": result["predicted_label"],
    "risk_level": result["risk_level"],
    "recommended_action": result["recommended_action"],
}

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
        return result
    

    except HTTPException:
        raise
    except Exception as e:
        print("SCORING ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"message": "Banking Fraud System API is running"}


# =========================================================
# BANKING SYSTEM ENDPOINTS
# =========================================================
@app.post("/accounts")
def create_account(request: CreateAccountRequest):
    account = bank.create_account(request.name, request.initial_balance)
    return account.to_dict()


@app.get("/accounts/{account_id}")
def get_account(account_id: str):
    try:
        account = bank.get_account(account_id)
        return account.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/deposit")
def deposit(request: DepositWithdrawRequest):
    try:
        account = bank.get_account(request.account_id)
        account.deposit(request.amount)
        return {
            "account_id": account.account_id,
            "new_balance": account.balance,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/withdraw")
def withdraw(request: DepositWithdrawRequest):
    try:
        account = bank.get_account(request.account_id)
        account.withdraw(request.amount)
        return {
            "account_id": account.account_id,
            "new_balance": account.balance,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/transfer")
def transfer(request: TransferRequest):
    try:
        bank.transfer(request.from_account_id, request.to_account_id, request.amount)
        return {"message": "Transfer successful"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================
# FRAUD SCORING ENDPOINTS
# =========================================================
@app.post("/predict_fraud")
def predict_fraud(request: FraudTransactionRequest):
    result = score_transaction(request.model_dump())
    return clean_for_json(result)


@app.post("/predict_fraud_csv")
async def predict_fraud_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        results = []
        for _, row in df.iterrows():
            payload = row[feature_columns].to_dict()
            results.append(score_transaction(payload))

        output_df = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(results)],
            axis=1,
        )

        output_df.to_csv(SCORED_FILE, index=False)
        result = {
            "message": "Batch scoring completed",
            "rows_scored": len(output_df),
            "output_file": str(SCORED_FILE),
            "preview": output_df.head(10).to_dict(orient="records"),
        }


        return clean_for_json(result)
    except HTTPException:
        raise
    except Exception as e:
        print("BATCH ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_report")
def download_report():
    if not SCORED_FILE.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(SCORED_FILE, filename="scored_transactions.csv")


# =========================================================
# MONITORING ENDPOINTS
# =========================================================
@app.get("/metrics")
def metrics():
    if not LOG_FILE.exists():
        return {
            "total_predictions": 0,
            "avg_probability": 0.0,
            "fraud_rate": 0.0,
        }

    try:
        df = pd.read_json(LOG_FILE, lines=True)

        return {
            "total_predictions": int(len(df)),
            "avg_probability": float(df["probability"].mean()) if len(df) else 0.0,
            "fraud_rate": float((df["prediction"] == 1).mean()) if len(df) else 0.0,
        }
    except Exception as e:
        print("METRICS ERROR:", e)
        return {
            "total_predictions": 0,
            "avg_probability": 0.0,
            "fraud_rate": 0.0,
        }


@app.get("/monitoring_data")
def monitoring_data():
    if not LOG_FILE.exists():
        return {"records": []}

    try:
        df = pd.read_json(LOG_FILE, lines=True)
        if df.empty:
            return {"records": []}

        df = df.tail(500)
        df= df.replace([np.nan, np.inf, -np.inf], 0)
        return {"records": df.to_dict(orient="records")}
    except Exception as e:
        print("MONITORING ERROR:", e)
        return {"records": []}
@app.get("/customers")
def customers():
    if not LOG_FILE.exists():
        return {"customers": []}

    try:
        df = pd.read_json(LOG_FILE, lines=True)

        if df.empty or "customer_id" not in df.columns:
            return {"customers": []}

        df = df.replace([np.nan, np.inf, -np.inf], 0)

        summary = (
            df.groupby("customer_id")
            .agg(
                total_transactions=("customer_id", "size"),
                avg_fraud_probability=("fraud_probability", "mean"),
                fraud_rate=("fraud_prediction", "mean"),
                last_action=("recommended_action", "last"),
            )
            .reset_index()
            .sort_values("avg_fraud_probability", ascending=False)
        )

        return {"customers": summary.to_dict(orient="records")}

    except Exception as e:
        print("CUSTOMERS ERROR:", e)
        return {"customers": []}