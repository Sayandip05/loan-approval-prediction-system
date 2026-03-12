@echo off
REM ========================================
REM Start All Services
REM ========================================

echo.
echo ========================================
echo  Starting Loan Approval Prediction System
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Check if model exists
if not exist "models\model.pkl" (
    echo WARNING: models\model.pkl not found.
    echo The API will start but predictions will return 503.
    echo Run the notebook first to train and save the model.
    echo.
)

echo.
echo Services will be available at:
echo   - Streamlit UI:  http://localhost:8501
echo   - FastAPI:       http://localhost:8000/docs
echo.

REM Start FastAPI in background
start "FastAPI" cmd /c "venv\Scripts\activate.bat && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start Streamlit (main window)
echo.
echo All services starting...
echo.
echo Press Ctrl+C in each window to stop services
echo.
streamlit run frontend\streamlit_app.py

pause
