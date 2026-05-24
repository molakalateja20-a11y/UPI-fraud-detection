# -*- coding: utf-8 -*-
"""
data_preparation.py
-------------------
Generates 5000 synthetic UPI transactions with realistic fraud patterns.
Outputs: data/transactions.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

# ─── Config ─────────────────────────────────────────────────────────────────
N_TOTAL       = 5000
FRAUD_RATIO   = 0.05          # 5 % fraud — realistic class imbalance
N_FRAUD       = int(N_TOTAL * FRAUD_RATIO)
N_LEGIT       = N_TOTAL - N_FRAUD

CITIES        = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
                 "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
TX_TYPES      = ["P2P", "P2M", "QR", "Auto-Pay", "Collect"]
BANKS         = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB"]

START_DATE    = datetime(2024, 1, 1)
END_DATE      = datetime(2024, 12, 31)
DATE_RANGE    = (END_DATE - START_DATE).days


def random_timestamp(night_bias=False):
    """Generate a random datetime; optionally bias toward night hours (10 PM – 6 AM)."""
    day    = int(np.random.randint(0, DATE_RANGE))
    if night_bias and np.random.random() < 0.75:
        hour   = int(np.random.choice(list(range(22, 24)) + list(range(0, 7))))
    else:
        hour   = int(np.random.randint(8, 21))
    minute = int(np.random.randint(0, 60))
    second = int(np.random.randint(0, 60))
    return START_DATE + timedelta(days=day, hours=hour, minutes=minute, seconds=second)


def velocity_score(hour, tx_type):
    """Simulate how many transactions came from same device in last hour (higher = suspicious)."""
    base = np.random.randint(1, 4)
    if tx_type in ["P2P", "QR"] and hour in list(range(22, 24)) + list(range(0, 7)):
        return base + np.random.randint(3, 10)
    return base


# ─── Generate Legitimate Transactions ───────────────────────────────────────
def generate_legitimate(n):
    records = []
    for _ in range(n):
        ts       = random_timestamp(night_bias=False)
        hour     = ts.hour
        amount   = np.random.choice([
            np.random.exponential(scale=500),          # small everyday
            np.random.uniform(1000, 10000),            # medium
            np.random.uniform(10000, 25000)            # large but not suspicious
        ], p=[0.70, 0.25, 0.05])
        amount   = round(max(50, min(amount, 24999)), 2)
        records.append({
            "transaction_id"   : f"TXN{np.random.randint(100000000, 999999999)}",
            "amount"           : amount,
            "transaction_type" : np.random.choice(TX_TYPES, p=[0.40, 0.25, 0.20, 0.10, 0.05]),
            "sender_bank"      : np.random.choice(BANKS),
            "receiver_bank"    : np.random.choice(BANKS),
            "location"         : np.random.choice(CITIES),
            "timestamp"        : ts.isoformat(),
            "hour"             : hour,
            "day_of_week"      : ts.weekday(),
            "is_night"         : int(hour >= 22 or hour < 6),
            "device_score"     : np.random.uniform(10, 45),   # low risk device
            "velocity"         : velocity_score(hour, "P2P"),
            "is_new_device"    : int(np.random.random() < 0.05),
            "is_round_amount"  : int(amount % 1000 == 0),
            "location_change"  : int(np.random.random() < 0.05),
            "is_fraud"         : 0
        })
    return records


# ─── Generate Fraudulent Transactions ───────────────────────────────────────
def generate_fraud(n):
    records = []
    for _ in range(n):
        ts       = random_timestamp(night_bias=True)
        hour     = ts.hour

        # Fraud patterns: high amounts, round numbers, or rapid small transfers
        pattern = np.random.choice(["high_amount", "night_transfer", "rapid_small"], p=[0.50, 0.35, 0.15])

        if pattern == "high_amount":
            amount = np.random.choice([25000, 30000, 35000, 40000, 45000, 50000])
        elif pattern == "night_transfer":
            amount = np.random.choice([10000, 15000, 20000, 25000])
        else:
            amount = round(np.random.uniform(500, 2000), 2)

        records.append({
            "transaction_id"   : f"TXN{np.random.randint(100000000, 999999999)}",
            "amount"           : float(amount),
            "transaction_type" : np.random.choice(TX_TYPES, p=[0.55, 0.15, 0.20, 0.05, 0.05]),
            "sender_bank"      : np.random.choice(BANKS),
            "receiver_bank"    : np.random.choice(BANKS),
            "location"         : np.random.choice(CITIES),
            "timestamp"        : ts.isoformat(),
            "hour"             : hour,
            "day_of_week"      : ts.weekday(),
            "is_night"         : int(hour >= 22 or hour < 6),
            "device_score"     : np.random.uniform(65, 100),  # high risk device
            "velocity"         : velocity_score(hour, "P2P") + np.random.randint(3, 8),
            "is_new_device"    : int(np.random.random() < 0.70),
            "is_round_amount"  : int(float(amount) % 1000 == 0),
            "location_change"  : int(np.random.random() < 0.60),
            "is_fraud"         : 1
        })
    return records


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  UPI Fraud Detection — Data Preparation")
    print("=" * 60)

    print(f"\n[DATA] Generating {N_LEGIT:,} legitimate transactions...")
    legit  = generate_legitimate(N_LEGIT)
    print(f"[DATA] Generating {N_FRAUD:,} fraudulent transactions...")
    fraud  = generate_fraud(N_FRAUD)

    df = pd.DataFrame(legit + fraud).sample(frac=1, random_state=42).reset_index(drop=True)

    # Derived features
    df["amount_log"]         = np.log1p(df["amount"])
    df["amount_anomaly"]     = ((df["amount"] - df["amount"].mean()) / df["amount"].std()).abs()
    df["hour_sin"]           = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]           = np.cos(2 * np.pi * df["hour"] / 24)
    df["risk_score"]         = (
        df["device_score"] * 0.40 +
        df["velocity"]     * 2.5  +
        df["is_night"]     * 15   +
        df["is_new_device"] * 20  +
        df["location_change"] * 10
    )

    # Save
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "transactions.csv")
    df.to_csv(out_path, index=False)

    print(f"\n[OK] Dataset saved -> {out_path}")
    print(f"     Total rows : {len(df):,}")
    print(f"     Fraud rows : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"     Features   : {df.shape[1]}")
    print("\nSample rows:")
    print(df[["amount", "hour", "device_score", "is_night", "is_fraud"]].head(10).to_string(index=False))
    print("\n[OK] Data preparation complete!")


if __name__ == "__main__":
    main()
