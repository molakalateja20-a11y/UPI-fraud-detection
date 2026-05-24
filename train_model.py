# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Trains a Random Forest classifier on the generated UPI transaction data.
Outputs: models/fraud_model.pkl, models/feature_scaler.pkl, models/model_metrics.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)

# ─── Feature columns used by the model ───────────────────────────────────────
FEATURE_COLS = [
    "amount", "amount_log", "amount_anomaly",
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "is_night",
    "device_score", "velocity",
    "is_new_device", "is_round_amount", "location_change",
    "risk_score"
]

TX_TYPE_MAP  = {"P2P": 0, "P2M": 1, "QR": 2, "Auto-Pay": 3, "Collect": 4}
LOCATION_MAP = {c: i for i, c in enumerate(
    ["Delhi","Mumbai","Bangalore","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad","Jaipur","Lucknow"]
)}


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_type"] = df["transaction_type"].map(TX_TYPE_MAP).fillna(0)
    df["location"]         = df["location"].map(LOCATION_MAP).fillna(0)
    return df


def main():
    print("=" * 60)
    print("  UPI Fraud Detection — Model Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    data_path = os.path.join("data", "transactions.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Please run  python data_preparation.py  first."
        )

    df = pd.read_csv(data_path)
    print(f"\n[DATA] Loaded {len(df):,} transactions  |  Fraud rate: {df['is_fraud'].mean()*100:.1f}%")

    df = encode_df(df)

    # Add tx_type and location to feature cols
    ALL_FEATURES = ["transaction_type", "location"] + FEATURE_COLS

    X = df[ALL_FEATURES]
    y = df["is_fraud"]

    # ── Train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train Random Forest ──────────────────────────────────────────────────
    print("\n[MODEL] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 10,
        min_samples_split = 5,
        class_weight  = "balanced",
        random_state  = 42,
        n_jobs        = -1
    )
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc         = accuracy_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred, zero_division=0)
    rec         = recall_score(y_test, y_pred, zero_division=0)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*40}")
    print(f"  Accuracy          : {acc*100:.1f}%")
    print(f"  Precision (Fraud) : {prec*100:.1f}%")
    print(f"  Recall (Fraud)    : {rec*100:.1f}%")
    print(f"  F1-Score          : {f1*100:.1f}%")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"{'─'*40}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'])}")
    print(f"Confusion Matrix:\n{cm}\n")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print("Top 10 Feature Importances:")
    for feat, imp in fi.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  os.path.join("models", "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "feature_scaler.pkl"))

    metrics = {
        "accuracy"          : round(acc * 100, 2),
        "precision_fraud"   : round(prec * 100, 2),
        "recall_fraud"      : round(rec * 100, 2),
        "f1_score"          : round(f1 * 100, 2),
        "roc_auc"           : round(roc_auc, 4),
        "confusion_matrix"  : cm.tolist(),
        "feature_importance": fi.round(4).to_dict(),
        "n_estimators"      : 100,
        "train_samples"     : len(X_train),
        "test_samples"      : len(X_test),
        "feature_columns"   : ALL_FEATURES
    }
    with open(os.path.join("models", "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[OK] Model saved  -> models/fraud_model.pkl")
    print("[OK] Scaler saved -> models/feature_scaler.pkl")
    print("[OK] Metrics saved-> models/model_metrics.json")
    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
