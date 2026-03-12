"""
Prediction module
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Union


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelNotLoadedError(Exception):
    """Raised when prediction is attempted but no model is loaded."""
    pass


class LoanDefaultPredictor:
    """Loan Approval Prediction Class"""

    REQUIRED_COLUMNS = [
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

    def __init__(self, model_path: str = None):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model
        """
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "model.pkl"

        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize model file '{self.model_path}': {e}")

    def _ensure_model_loaded(self):
        """Guard that raises ModelNotLoadedError if model is None."""
        if self.model is None:
            raise ModelNotLoadedError("No model is loaded. Train and save a model first.")

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

    def preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data

        Args:
            data: Input data as dict or DataFrame

        Returns:
            Preprocessed DataFrame
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Expected dict or DataFrame, got {type(data).__name__}")

        self._validate_input(df)

        # Apply same feature engineering as training
        df = self._engineer_features(df)

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering (same as training)"""

        # Debt to income ratio
        df['DebtToIncomeRatio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)

        # Credit utilization category
        df['CreditUtilization_Category'] = pd.cut(
            df['RevolvingUtilizationOfUnsecuredLines'],
            bins=[-np.inf, 0.3, 0.6, 1.0, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Age group
        df['AgeGroup'] = pd.cut(
            df['age'],
            bins=[-np.inf, 30, 45, 60, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Total past due
        df['TotalPastDue'] = (
            df['NumberOfTime30-59DaysPastDueNotWorse'] +
            df['NumberOfTime60-89DaysPastDueNotWorse'] +
            df['NumberOfTimes90DaysLate']
        )

        # Has past due
        df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)

        # Income per dependent
        df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

        # Log monthly income
        df['LogMonthlyIncome'] = np.log1p(df['MonthlyIncome'])

        # Loans per credit line
        df['LoansPerCreditLine'] = df['NumberRealEstateLoansOrLines'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)

        return df

    def predict(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions

        Args:
            data: Input data

        Returns:
            Predictions (0 or 1)
        """
        self._ensure_model_loaded()
        df = self.preprocess_input(data)
        predictions = self.model.predict(df)
        return predictions

    def predict_proba(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Predict probabilities

        Args:
            data: Input data

        Returns:
            Prediction probabilities
        """
        self._ensure_model_loaded()
        df = self.preprocess_input(data)
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Loaded model does not support predict_proba")
        probabilities = self.model.predict_proba(df)
        return probabilities

    def predict_single(self, data: Dict) -> Dict:
        """
        Predict for single input and return detailed result

        Args:
            data: Single input as dictionary

        Returns:
            Dictionary with prediction and probability
        """
        if not isinstance(data, dict):
            raise TypeError(f"predict_single expects a dict, got {type(data).__name__}")

        prediction = self.predict(data)[0]
        proba = self.predict_proba(data)[0]

        return {
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default',
            'probability_no_default': float(proba[0]),
            'probability_default': float(proba[1]),
            'risk_level': self._get_risk_level(proba[1])
        }

    def _get_risk_level(self, default_prob: float) -> str:
        """Categorize risk level based on default probability"""
        if default_prob < 0.3:
            return 'Low Risk'
        elif default_prob < 0.6:
            return 'Medium Risk'
        else:
            return 'High Risk'

    def explain(self, data: Dict) -> Dict:
        """
        Generate SHAP feature importance for a single prediction.

        Args:
            data: Single input as dictionary

        Returns:
            Dictionary with feature names and their SHAP values
        """
        self._ensure_model_loaded()

        try:
            import shap
        except ImportError:
            raise ImportError("shap package is required for explanations. Install with: pip install shap")

        df = self.preprocess_input(data)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(df)

        # For binary classifiers shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # SHAP values for class 1 (default)
        else:
            sv = shap_values[0]

        feature_names = df.columns.tolist()
        contributions = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            "features": [c[0] for c in contributions],
            "shap_values": [round(c[1], 6) for c in contributions],
            "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
        }


if __name__ == "__main__":
    try:
        # Test prediction
        predictor = LoanDefaultPredictor()

        # Sample input
        sample_data = {
            'RevolvingUtilizationOfUnsecuredLines': 0.766127,
            'age': 45,
            'NumberOfTime30-59DaysPastDueNotWorse': 2,
            'DebtRatio': 0.802982,
            'MonthlyIncome': 9120,
            'NumberOfOpenCreditLinesAndLoans': 13,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 6,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 2
        }

        result = predictor.predict_single(sample_data)
        print("\nPrediction Result:")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Default Probability: {result['probability_default']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error("Run the notebook (notebooks/01_model_development.ipynb) first to train the model.")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise SystemExit(1)
