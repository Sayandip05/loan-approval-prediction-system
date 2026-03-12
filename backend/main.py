"""
FastAPI Application for Loan Approval Prediction System
"""
import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import io

from .model.predict import LoanDefaultPredictor, ModelNotLoadedError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Initialize predictor lazily to avoid crashing the app on import
predictor: Optional[LoanDefaultPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the application starts."""
    global predictor
    try:
        predictor = LoanDefaultPredictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor on startup: {e}")
        predictor = None
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval probability",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_predictor() -> LoanDefaultPredictor:
    """Return the predictor or raise a clear HTTP error if unavailable."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train and save a model first, then restart the server."
        )
    return predictor


# Pydantic models for request/response
class LoanInput(BaseModel):
    """Input schema for single prediction"""
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=50, description="Credit utilization ratio (typically 0-1, capped at 50)")
    age: int = Field(..., ge=18, le=120, description="Age of borrower (18-120)")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, le=98, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0, le=50000, description="Debt ratio")
    MonthlyIncome: float = Field(..., ge=0, le=500000, description="Monthly income in dollars")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, le=100, description="Number of open credit lines")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, le=98, description="Number of times 90+ days late")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, le=50, description="Number of real estate loans")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, le=98, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(..., ge=0, le=20, description="Number of dependents")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.766127,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.802982,
                "MonthlyIncome": 9120,
                "NumberOfOpenCreditLinesAndLoans": 13,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 6,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    prediction: int
    prediction_label: str
    probability_no_default: float
    probability_default: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Output schema for batch prediction"""
    total_predictions: int
    predictions: List[Dict]


MAX_BATCH_ROWS = 10_000


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Loan Approval Prediction API is running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    model_loaded = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": str(predictor.model_path) if predictor else "N/A"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(loan_data: LoanInput):
    """
    Make a single prediction

    Args:
        loan_data: Loan applicant data

    Returns:
        Prediction result with probability
    """
    pred = _get_predictor()
    try:
        # Convert to dict
        data_dict = loan_data.model_dump(by_alias=True)

        # Make prediction
        result = pred.predict_single(data_dict)

        return PredictionResponse(**result)

    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Make batch predictions from CSV file

    Args:
        file: CSV file with loan data

    Returns:
        Batch prediction results
    """
    pred = _get_predictor()

    # Validate file type
    if file.content_type not in (None, "text/csv", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a CSV file, got content-type '{file.content_type}'"
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        decoded = contents.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text")

    try:
        df = pd.read_csv(io.StringIO(decoded))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file has no data")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file produced an empty DataFrame")

    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(df)} exceeds maximum of {MAX_BATCH_ROWS} rows"
        )

    # Validate required columns
    required_columns = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing_cols)}"
        )

    # Check for NaN in required columns
    nan_counts = df[required_columns].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Columns with missing values: {cols_with_nan.to_dict()}"
        )

    try:
        predictions = pred.predict(df)
        probabilities = pred.predict_proba(df)
    except Exception as e:
        logger.error(f"Batch prediction model error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    # Format results
    results = []
    for idx, (p, proba) in enumerate(zip(predictions, probabilities)):
        results.append({
            'id': idx,
            'prediction': int(p),
            'prediction_label': 'Default' if p == 1 else 'No Default',
            'probability_no_default': float(proba[0]),
            'probability_default': float(proba[1]),
            'risk_level': pred._get_risk_level(proba[1])
        })

    return BatchPredictionResponse(
        total_predictions=len(results),
        predictions=results
    )


@app.get("/model_info", tags=["Model"])
async def model_info():
    """Get model information"""
    pred = _get_predictor()
    try:
        info = {
            "model_type": type(pred.model).__name__,
            "model_path": str(pred.model_path),
            "features_count": len(pred.model.feature_names_in_) if hasattr(pred.model, 'feature_names_in_') else "N/A"
        }
        # Include metrics if available
        metrics_path = PROJECT_ROOT / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                info["metrics"] = metrics
            except (json.JSONDecodeError, IOError):
                info["metrics"] = None
        else:
            info["metrics"] = None
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.get("/metrics", tags=["Model"])
async def get_metrics():
    """Get training metrics from metrics.json"""
    metrics_path = PROJECT_ROOT / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Train the model first.")
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="metrics.json is corrupted")
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics.json: {e}")


@app.post("/explain", tags=["Explainability"])
async def explain_prediction(loan_data: LoanInput):
    """
    Get SHAP feature importance for a single prediction.

    Returns the SHAP values showing how each feature contributed
    to pushing the prediction towards default or no-default.
    """
    pred = _get_predictor()
    try:
        data_dict = loan_data.model_dump(by_alias=True)
        explanation = pred.explain(data_dict)
        return explanation
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Explain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
