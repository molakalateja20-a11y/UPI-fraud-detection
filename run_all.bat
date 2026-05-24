@echo off
title UPI Fraud Detection System - Launcher
color 0A

echo ============================================================
echo   UPI FRAUD DETECTION SYSTEM - AUTO SETUP ^& LAUNCH
echo ============================================================
echo.

:: Step 1: Install dependencies
echo [1/4] Installing Python dependencies ...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERROR: pip install failed. Please check your Python installation.
    pause
    exit /b 1
)
echo        Done!
echo.

:: Step 2: Generate data
echo [2/4] Generating synthetic UPI transaction data ...
python data_preparation.py
if %errorlevel% neq 0 (
    echo ERROR: Data preparation failed.
    pause
    exit /b 1
)
echo.

:: Step 3: Train model
echo [3/4] Training Random Forest model ...
python train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)
echo.

:: Step 4: Launch both services
echo [4/4] Launching API Server and Dashboard ...
echo.
echo  API Server  -> http://127.0.0.1:5000
echo  Dashboard   -> http://127.0.0.1:8501
echo.
echo Press Ctrl+C in each window to stop the services.
echo ============================================================

:: Start API server in a new window
start "UPI Fraud - API Server" cmd /k python api_server.py

:: Wait 3 seconds for API to start
timeout /t 3 /nobreak >nul

:: Start Streamlit dashboard
start "UPI Fraud - Dashboard" cmd /k python -m streamlit run dashboard.py --server.port 8501

echo.
echo Both services are starting...
echo Open http://127.0.0.1:8501 in your browser.
echo.
pause
