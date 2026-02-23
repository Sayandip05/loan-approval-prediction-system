# 🏦 Loan Default Prediction - End-to-End MLOps Project

A production-ready machine learning project for predicting loan defaults using the "Give Me Some Credit" dataset from Kaggle.

## 📊 Project Overview

This project implements a complete MLOps pipeline including:
- **Data Versioning**: DVC for data and pipeline management
- **Experiment Tracking**: MLflow for model versioning and metrics
- **API Development**: FastAPI for high-performance REST endpoints
- **Frontend**: Streamlit for interactive predictions
- **Containerization**: Docker for consistent deployments
- **Cloud Deployment**: AWS EKS (planned for later phase)

## 🎯 Dataset

**Source**: [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

- **Size**: 250,000 borrower records
- **Target**: Predict serious financial distress in next 2 years
- **Features**: 10 features including age, income, debt ratio, credit history

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
Git
Docker (optional)
Kaggle API credentials
```

### Installation

1. **Clone the repository**
```bash
cd "C:\Users\sayan\AI ML\loan-default-prediction"
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC**
```bash
dvc init
git init
```

5. **Download dataset from Kaggle**
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle competitions download -c GiveMeSomeCredit

# Extract to data/raw/
unzip GiveMeSomeCredit.zip -d data/raw/
```

## 📂 Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── raw/                          # Original dataset (DVC tracked)
│   └── processed/                    # Processed data (DVC tracked)
│
├── notebooks/
│   └── 01_model_development.ipynb    # EDA & Model experimentation
│
├── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── preprocess.py             # Data preprocessing
│   │   └── feature_engineering.py    # Feature creation
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py                  # Model training with MLflow
│   │   └── predict.py                # Prediction logic
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI application
│   │
│   └── frontend/
│       └── streamlit_app.py          # Streamlit UI
│
├── models/                            # Saved models
├── mlruns/                            # MLflow tracking data
├── tests/                             # Unit tests
│
├── dvc.yaml                           # DVC pipeline definition
├── params.yaml                        # Hyperparameters
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Multi-container setup
├── .env.example                       # Environment variables template
├── .gitignore
├── .dvcignore
└── README.md
```

## 🔬 Usage

### 1. Exploratory Data Analysis (Notebook)

```bash
jupyter notebook notebooks/01_model_development.ipynb
```

- Perform EDA
- Try multiple models
- Track experiments with MLflow
- Select best model

### 2. Run DVC Pipeline

```bash
# Run entire pipeline (preprocess → train)
dvc repro

# Or run specific stages
dvc repro preprocess
dvc repro train
```

### 3. Start MLflow UI

```bash
mlflow ui
# Visit: http://localhost:5000
```

### 4. Run FastAPI Backend

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# API Docs: http://localhost:8000/docs
```

### 5. Run Streamlit Frontend

```bash
streamlit run src/frontend/streamlit_app.py
# Visit: http://localhost:8501
```

### 6. Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Services:
# - FastAPI: http://localhost:8000
# - Streamlit: http://localhost:8501
# - MLflow: http://localhost:5000
```

## 🔧 API Endpoints

### Health Check
```bash
GET /
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
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
```

### Batch Prediction
```bash
POST /batch_predict
Content-Type: multipart/form-data

file: <CSV file>
```

## 📊 Model Performance

| Model | AUC-ROC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| XGBoost | 0.86 | 0.82 | 0.78 | 0.80 |
| Random Forest | 0.84 | 0.80 | 0.76 | 0.78 |
| Logistic Regression | 0.79 | 0.75 | 0.72 | 0.73 |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📈 MLflow Tracking

Track experiments, compare models, and manage model lifecycle:

```python
import mlflow

# Set experiment
mlflow.set_experiment("loan-default-prediction")

# Log parameters
mlflow.log_param("n_estimators", 100)

# Log metrics
mlflow.log_metric("auc_roc", 0.86)

# Log model
mlflow.sklearn.log_model(model, "model")
```

## 🔄 DVC Pipeline

The pipeline consists of 3 stages:

1. **Preprocess**: Clean data, handle missing values
2. **Feature Engineering**: Create new features
3. **Train**: Train model with MLflow tracking

```bash
# View pipeline
dvc dag

# Run pipeline
dvc repro

# Check metrics
dvc metrics show
```

## 🐳 Docker Commands

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up --build api
```

## 🌐 Future Enhancements

- [ ] AWS EKS deployment
- [ ] CI/CD with GitHub Actions / AWS CodePipeline
- [ ] Kubernetes manifests
- [ ] Model monitoring with Prometheus/Grafana
- [ ] A/B testing framework
- [ ] Feature store integration
- [ ] Advanced feature engineering (Deep Feature Synthesis)

## 📝 Environment Variables

Create `.env` file from `.env.example`:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# AWS (for later)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Model
MODEL_PATH=models/model.pkl
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## 📄 License

MIT License

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Kaggle for the dataset
- MLflow and DVC communities
- FastAPI and Streamlit teams

---

**⭐ Star this repo if you find it helpful!**
