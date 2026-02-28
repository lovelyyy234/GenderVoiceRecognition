@echo off
title Gender Voice Recognition V3 - Setup
color 0B

echo ============================================================
echo   Gender Voice Recognition V3  -  Web App Edition
echo ============================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

echo [1/5] Python detected:
python --version
echo.

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Done.
echo.

echo [3/5] Installing core packages...
pip install flask werkzeug numpy pandas scipy scikit-learn xgboost librosa soundfile --quiet
echo Done.
echo.

echo [4/5] Installing TensorFlow (CNN + LSTM support)...
echo This may take 2-5 minutes...
pip install tensorflow --quiet
if %errorlevel% neq 0 (
    echo [WARN] TensorFlow install failed. CNN and LSTM will be disabled.
    echo         The app still works with ML models (SVM/RF/XGBoost).
)
echo Done.
echo.

echo [5/5] Starting web application...
echo.
echo ============================================================
echo   Open your browser at:  http://127.0.0.1:5000
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed. Check error above.
    pause
)
