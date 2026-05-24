# -*- coding: utf-8 -*-
"""
api_server.py
-------------
Flask REST API for real-time UPI fraud detection.

Endpoints:
  GET  /health          – Server health check
  GET  /stats           – Model statistics
  POST /predict         – Single transaction prediction
  POST /batch           – Batch transaction prediction
"""

import os
import json
import joblib
import sqlite3
import random
import numpy as np
import pandas as pd
from datetime  import datetime
from flask     import Flask, request, jsonify
from flask_cors import CORS


# ─── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "fraud_model.pkl")
SCALER_PATH  = os.path.join("models", "feature_scaler.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please run  python train_model.py  first."
    )

model   = joblib.load(MODEL_PATH)
scaler  = joblib.load(SCALER_PATH)

with open(METRICS_PATH) as f:
    MODEL_METRICS = json.load(f)

FEATURE_COLUMNS = MODEL_METRICS["feature_columns"]

# ─── Encoding Maps ───────────────────────────────────────────────────────────
TX_TYPE_MAP  = {"P2P": 0, "P2M": 1, "QR": 2, "Auto-Pay": 3, "Collect": 4}
LOCATION_MAP = {c: i for i, c in enumerate(
    ["Delhi","Mumbai","Bangalore","Hyderabad","Chennai",
     "Kolkata","Pune","Ahmedabad","Jaipur","Lucknow"]
)}

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

REQUEST_LOG = []   # in-memory log for demo purposes

DB_PATH = os.path.join("data", "transactions.db")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            timestamp TEXT,
            amount REAL,
            transaction_type TEXT,
            location TEXT,
            hour INTEGER,
            device_score REAL,
            velocity INTEGER,
            is_new_device INTEGER,
            location_change INTEGER,
            fraud_probability REAL,
            risk_level TEXT,
            action TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_transaction(tx_data: dict, prob: float, risk: str, action: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tx_id = tx_data.get("transaction_id") or f"TXN{random.randint(1000000000, 9999999999)}"
    ts = tx_data.get("timestamp") or datetime.now().isoformat()
    amount = float(tx_data.get("amount", 0))
    tx_type = tx_data.get("transaction_type", "P2P")
    loc = tx_data.get("location", "Delhi")
    
    try:
        ts_parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        hour = ts_parsed.hour
    except Exception:
        hour = int(tx_data.get("hour", datetime.now().hour))
        
    dev_score = float(tx_data.get("device_score", 30.0))
    vel = int(tx_data.get("velocity", 1))
    new_device = int(tx_data.get("is_new_device", 0))
    loc_change = int(tx_data.get("location_change", 0))
    
    cursor.execute("""
        INSERT INTO transactions (
            transaction_id, timestamp, amount, transaction_type, location,
            hour, device_score, velocity, is_new_device, location_change,
            fraud_probability, risk_level, action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tx_id, ts, amount, tx_type, loc, hour, dev_score, vel,
        new_device, loc_change, round(prob, 4), risk, action
    ))
    
    conn.commit()
    conn.close()
    return tx_id



# ─── Feature Engineering ─────────────────────────────────────────────────────
def parse_transaction(data: dict) -> pd.DataFrame:
    """Convert raw API payload to model-ready feature vector."""
    # Parse timestamp
    ts_str = data.get("timestamp", datetime.now().isoformat())
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        ts = datetime.now()

    hour        = ts.hour
    day_of_week = ts.weekday()
    is_night    = int(hour >= 22 or hour < 6)

    amount      = float(data.get("amount", 0))
    amount_log  = np.log1p(amount)

    # For anomaly score we use a rough reference (mean=5000, std=8000)
    REF_MEAN, REF_STD = 5000, 8000
    amount_anomaly = abs((amount - REF_MEAN) / REF_STD)

    device_score    = float(data.get("device_score", 30))
    velocity        = int(data.get("velocity", 2))
    is_new_device   = int(data.get("is_new_device", 0))
    is_round_amount = int(amount % 1000 == 0)
    location_change = int(data.get("location_change", 0))

    risk_score = (
        device_score * 0.40 +
        velocity     * 2.5  +
        is_night     * 15   +
        is_new_device * 20  +
        location_change * 10
    )

    tx_type  = TX_TYPE_MAP.get(data.get("transaction_type", "P2P"), 0)
    location = LOCATION_MAP.get(data.get("location", "Delhi"), 0)

    row = {
        "transaction_type" : tx_type,
        "location"         : location,
        "amount"           : amount,
        "amount_log"       : amount_log,
        "amount_anomaly"   : amount_anomaly,
        "hour"             : hour,
        "hour_sin"         : np.sin(2 * np.pi * hour / 24),
        "hour_cos"         : np.cos(2 * np.pi * hour / 24),
        "day_of_week"      : day_of_week,
        "is_night"         : is_night,
        "device_score"     : device_score,
        "velocity"         : velocity,
        "is_new_device"    : is_new_device,
        "is_round_amount"  : is_round_amount,
        "location_change"  : location_change,
        "risk_score"       : risk_score,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def classify(prob: float) -> dict:
    """Map probability to risk level, action, and alerts."""
    if prob >= 0.70:
        risk   = "HIGH"
        action = "BLOCK TRANSACTION"
        color  = "red"
    elif prob >= 0.40:
        risk   = "MEDIUM"
        action = "REQUIRE 2FA"
        color  = "orange"
    else:
        risk   = "LOW"
        action = "ALLOW TRANSACTION"
        color  = "green"
    return {"risk_level": risk, "action": action, "color": color}


def generate_alerts(data: dict, prob: float) -> list:
    alerts = []
    amount = float(data.get("amount", 0))
    ts_str = data.get("timestamp", datetime.now().isoformat())
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        hour = ts.hour
    except Exception:
        hour = datetime.now().hour

    if amount >= 25000:
        alerts.append("⚠️ High transaction amount (≥ ₹25,000)")
    if hour >= 22 or hour < 6:
        alerts.append("🌙 Transaction at unusual night hours")
    if data.get("is_new_device", 0):
        alerts.append("📱 Transaction from new/unrecognized device")
    if data.get("location_change", 0):
        alerts.append("📍 Unusual location — different from usual city")
    if data.get("device_score", 0) > 65:
        alerts.append("🔴 High device risk score")
    if data.get("velocity", 1) > 5:
        alerts.append("⚡ High transaction velocity (multiple in short time)")
    if amount % 1000 == 0 and amount >= 10000:
        alerts.append("🔢 Suspicious round-number amount")
    if not alerts and prob > 0.5:
        alerts.append("⚠️ Unusual transaction pattern detected by ML model")
    return alerts


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "<h2>🛡️ UPI FraudShield AI - REST API Server</h2><p>Status: 🟢 Active & Protecting</p><p>Endpoints: <a href='/health'>/health</a>, /stats, /predict</p>"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"    : "online",
        "model"     : "RandomForestClassifier",
        "version"   : "1.0.0",
        "timestamp" : datetime.now().isoformat()
    })


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "model_metrics"       : MODEL_METRICS,
        "total_requests"      : len(REQUEST_LOG),
        "fraud_flagged"       : sum(1 for r in REQUEST_LOG if r.get("risk_level") == "HIGH"),
        "server_start"        : SERVER_START
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        features = parse_transaction(data)
        scaled   = scaler.transform(features)
        prob     = float(model.predict_proba(scaled)[0][1])
        clf      = classify(prob)
        alerts   = generate_alerts(data, prob)

        tx_id = log_transaction(data, prob, clf["risk_level"], clf["action"])

        result = {
            "transaction_id"    : tx_id,
            "fraud_probability" : round(prob, 4),
            "risk_level"        : clf["risk_level"],
            "action"            : clf["action"],
            "color"             : clf["color"],
            "alerts"            : alerts,
            "timestamp"         : datetime.now().isoformat()
        }
        REQUEST_LOG.append(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch():
    payload = request.get_json(force=True)
    if not isinstance(payload, list):
        return jsonify({"error": "Payload must be a JSON array of transactions"}), 400

    results = []
    for tx in payload:
        try:
            features = parse_transaction(tx)
            scaled   = scaler.transform(features)
            prob     = float(model.predict_proba(scaled)[0][1])
            clf      = classify(prob)
            alerts   = generate_alerts(tx, prob)
            
            tx_id = log_transaction(tx, prob, clf["risk_level"], clf["action"])
            
            results.append({
                "transaction_id"    : tx_id,
                "fraud_probability" : round(prob, 4),
                "risk_level"        : clf["risk_level"],
                "action"            : clf["action"],
                "alerts"            : alerts
            })
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify({"results": results, "total": len(results)})


@app.route("/db_query", methods=["POST"])
def db_query():
    data = request.get_json(force=True) or {}
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "No SQL query provided"}), 400
        
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return jsonify({"error": "Only SELECT (read-only) queries are allowed for security."}), 403
        
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return jsonify({"success": True, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400



# ─── Main ────────────────────────────────────────────────────────────────────
SERVER_START = datetime.now().isoformat()

if __name__ == "__main__":
    print("=" * 60)
    print("  UPI Fraud Detection API Server")
    print("  Running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
