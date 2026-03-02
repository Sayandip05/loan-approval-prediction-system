"""
FastAPI Application for Loan Default Prediction
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import io

from .model.predict import LoanDefaultPredictor

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default probability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = LoanDefaultPredictor()


# Pydantic models for request/response
class LoanInput(BaseModel):
    """Input schema for single prediction"""
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description="Credit utilization ratio")
    age: int = Field(..., gt=0, le=120, description="Age of borrower")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0, description="Debt ratio")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Number of credit lines")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, description="Number of times 90+ days late")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description="Number of real estate loans")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(..., ge=0, description="Number of dependents")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
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


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Loan Default Prediction API is running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_path": str(predictor.model_path)
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
    try:
        # Convert to dict
        data_dict = loan_data.model_dump(by_alias=True)
        
        # Make prediction
        result = predictor.predict_single(data_dict)
        
        return PredictionResponse(**result)
    
    except Exception as e:
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
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
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
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Make predictions
        predictions = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        
        # Format results
        results = []
        for idx, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'id': idx,
                'prediction': int(pred),
                'prediction_label': 'Default' if pred == 1 else 'No Default',
                'probability_no_default': float(proba[0]),
                'probability_default': float(proba[1]),
                'risk_level': predictor._get_risk_level(proba[1])
            })
        
        return BatchPredictionResponse(
            total_predictions=len(results),
            predictions=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info", tags=["Model"])
async def model_info():
    """Get model information"""
    try:
        return {
            "model_type": type(predictor.model).__name__,
            "model_path": str(predictor.model_path),
            "features_count": len(predictor.model.feature_names_in_) if hasattr(predictor.model, 'feature_names_in_') else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
