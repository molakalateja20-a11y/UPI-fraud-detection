# 🛡️ UPI Fraud Detection System

A complete, end-to-end **Real-time UPI Fraud Detection** system using Machine Learning.

## 📁 Project Structure

```
upi_fraud_detection/
├── data_preparation.py       # Generate 5000 synthetic UPI transactions
├── train_model.py            # Train RandomForest model + save metrics
├── api_server.py             # Flask REST API (port 5000)
├── dashboard.py              # Streamlit web dashboard (port 8501)
├── requirements.txt          # Python dependencies
├── run_all.bat               # One-click setup & launch
├── models/                   # Saved model files (auto-created)
└── data/                     # CSV dataset (auto-created)
```

## ⚡ Quick Start

### Option A: One-Click (Windows)
```
Double-click run_all.bat
```

### Option B: Step-by-Step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python data_preparation.py

# 3. Train model
python train_model.py

# 4a. Start API server (keep this terminal open)
python api_server.py

# 4b. Start dashboard (new terminal)
streamlit run dashboard.py
```

Then open **http://127.0.0.1:8501** in your browser.

## 🔌 API Reference

### `POST /predict`
```json
{
  "amount": 45000,
  "transaction_type": "P2P",
  "location": "Delhi",
  "timestamp": "2024-01-15T02:30:00",
  "device_score": 85,
  "velocity": 8,
  "is_new_device": 1,
  "location_change": 1
}
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "risk_level": "HIGH",
  "action": "BLOCK TRANSACTION",
  "alerts": ["⚠️ High amount at night", "📱 New device detected"]
}
```

### `GET /health` — Server health check
### `GET /stats` — Model statistics
### `POST /batch` — Analyze multiple transactions

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~94.5% |
| Fraud Recall | ~65% |
| Precision | ~78% |
| ROC-AUC | ~0.92 |

## 🏗️ Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Model | RandomForestClassifier (scikit-learn) |
| API Server | Flask + Flask-CORS |
| Dashboard | Streamlit + Plotly |
| Data | Pandas + NumPy |

## 🎯 Risk Levels

| Risk | Threshold | Action |
|------|-----------|--------|
| 🟢 LOW | < 40% | Allow transaction |
| 🟡 MEDIUM | 40–70% | Require 2FA |
| 🔴 HIGH | > 70% | Block transaction |
