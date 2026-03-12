@echo off
REM ========================================
REM Loan Approval Prediction System - Setup Script
REM ========================================

echo.
echo ========================================
echo  Loan Approval Prediction System - Setup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo [1/4] Virtual environment already exists
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies...
pip install -r requirements.txt
echo ✅ Dependencies installed

echo.
echo [4/4] Setup complete!
echo.
echo ========================================
echo  Next Steps:
echo ========================================
echo.
echo 1. Download dataset from Kaggle:
echo    https://www.kaggle.com/c/GiveMeSomeCredit/data
echo.
echo 2. Place cs-training.csv in: data\raw\
echo.
echo 3. Open the notebook to train the model:
echo    jupyter notebook notebooks\01_model_development.ipynb
echo    (Run all cells - model saves to models\model.pkl)
echo.
echo 4. Start services: start_services.bat
echo.
echo ========================================

pause
