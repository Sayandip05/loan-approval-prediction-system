# Loan Default Prediction System

Predicts whether a borrower will experience serious financial distress within 2 years, using the **Give Me Some Credit** dataset from Kaggle.

The model is trained in a Jupyter notebook (locally or on Google Colab), then served through **FastAPI** (REST API) and **Streamlit** (web dashboard). Optionally deployable via **Docker**.

## Architecture

```
┌────────────────┐      HTTP       ┌────────────────┐
│   Streamlit    │ ──────────────► │    FastAPI     │
│   (Frontend)   │ ◄────────────── │    (Backend)   │
│  localhost:8501│                 │  localhost:8000│
└────────────────┘                 └───────┬────────┘
                                           │
                                    loads model.pkl
                                    reads metrics.json
                                           │
                                   ┌───────┴────────┐
                                   │  models/       │
                                   │   model.pkl    │
                                   │  metrics.json  │
                                   └────────────────┘
```

**Workflow:** Train in notebook → get `model.pkl` + `metrics.json` → serve with FastAPI → visualize with Streamlit

## Project Structure

```
Loan Approval Prediction System/
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app (7 endpoints)
│   └── model/
│       ├── __init__.py
│       └── predict.py              # Inference, feature engineering, SHAP
│
├── frontend/
│   ├── __init__.py
│   └── streamlit_app.py            # Streamlit dashboard
│
├── notebooks/
│   └── 01_model_development.ipynb  # EDA, training, model export
│
├── data/
│   └── raw/                        # Place cs-training.csv here
│
├── models/                         # model.pkl goes here (generated)
│
├── metrics.json                    # Training metrics (generated)
├── requirements.txt
├── Dockerfile                      # Multi-stage (api + frontend)
├── docker-compose.yml              # Two-container setup
├── .dockerignore
├── .env.example
├── .gitignore
├── setup.bat                       # Windows: create venv + install deps
└── start_services.bat              # Windows: launch FastAPI + Streamlit
```

## Dataset

**Source:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

- 250,000 borrower records
- Target: `SeriousDlqin2yrs` (1 = default, 0 = no default)
- 10 input features (age, income, debt ratio, credit lines, payment history, etc.)

## Quick Start

### Prerequisites

- Python 3.10+
- Kaggle account (to download the dataset)

### Step 1 — Train the model

**Option A: Google Colab (no local Python needed)**

1. Upload `notebooks/01_model_development.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `cs-training.csv` (from Kaggle) to Colab's file browser
3. Run all cells top to bottom
4. Download these two files from Colab and place them in the project:
   - `model.pkl` → `models/model.pkl`
   - `metrics.json` → project root

**Option B: Local Jupyter**

```bash
# Create virtual environment and install dependencies
setup.bat          # Windows
# Or manually:
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Place cs-training.csv in data/raw/, then:
jupyter notebook notebooks/01_model_development.ipynb
```

Run all cells. The notebook saves `models/model.pkl` and `metrics.json` automatically.

### Step 2 — Verify generated files

Before starting the app, confirm:

```
models/model.pkl       ← trained XGBoost model
metrics.json           ← training metrics (AUC, precision, recall, etc.)
```

- Without `model.pkl` the API starts but returns **503** on prediction endpoints.
- Without `metrics.json` the Model Info page shows "metrics unavailable" (not fatal).

### Step 3 — Run the app

```bash
# Activate virtual environment
venv\Scripts\activate

# Terminal 1 — FastAPI
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit
streamlit run frontend/streamlit_app.py
```

Or on Windows:
```bash
start_services.bat
```

| Service | URL |
|---------|-----|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI Swagger docs | http://localhost:8000/docs |
| API health check | http://localhost:8000/health |

## What the notebook does

`notebooks/01_model_development.ipynb` runs the full ML pipeline:

1. **EDA** — target distribution, correlations, missing values
2. **Preprocessing** — median imputation, outlier capping
3. **Feature engineering** — creates 8 derived features:
   - `DebtToIncomeRatio`, `CreditUtilization_Category`, `AgeGroup`
   - `TotalPastDue`, `HasPastDue`, `IncomePerDependent`
   - `LogMonthlyIncome`, `LoansPerCreditLine`
4. **SMOTE** — balances the 93:7 class imbalance
5. **Training** — compares Logistic Regression, Random Forest, XGBoost, LightGBM
6. **Evaluation** — ROC-AUC, precision, recall, F1, confusion matrix, ROC curve
7. **Export** — saves `model.pkl` and `metrics.json`

MLflow experiment tracking is included but optional — the notebook works with or without an MLflow server running.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Basic health check |
| GET | `/health` | Detailed health (model loaded status) |
| POST | `/predict` | Single loan prediction |
| POST | `/batch_predict` | Batch prediction via CSV upload |
| GET | `/model_info` | Model metadata + training metrics |
| GET | `/metrics` | Raw metrics from metrics.json |
| POST | `/explain` | SHAP feature contributions for one input |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Streamlit Dashboard Pages

| Page | What it shows |
|------|--------------|
| Home | Overview, quick stats, how-it-works |
| Single Prediction | Input form → prediction + probability gauge + SHAP bar chart |
| Batch Prediction | CSV upload → summary stats + pie/bar charts + downloadable results |
| Model Info | Model type, feature count, real training metrics, confusion matrix |

## Docker Deployment

Docker packages the API and frontend into two containers.

### How it works

The `Dockerfile` is a multi-stage build with two targets:

| Stage | Runs | Port |
|-------|------|------|
| `api` | FastAPI + Uvicorn | 8000 |
| `frontend` | Streamlit | 8501 |

`docker-compose.yml` connects them on an internal Docker network (`loan-network`). The frontend container has `API_URL=http://api:8000` set as an environment variable, so Streamlit calls FastAPI using Docker's internal DNS — `api` resolves to the API container's IP address.

The api container **mounts** `./models` and `./metrics.json` from your host, so you can update the model without rebuilding the image.

### Run with Docker

```bash
# Prerequisites: model.pkl must exist in models/
# and metrics.json must exist in project root

# Build and start
docker-compose up --build

# Or detached (background)
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs api
docker-compose logs frontend

# Stop
docker-compose down
```

Open http://localhost:8501 for the dashboard, http://localhost:8000/docs for the API.

### Docker troubleshooting

| Problem | Fix |
|---------|-----|
| API returns 503 on `/predict` | `model.pkl` missing — run the notebook first |
| Model Info shows "metrics unavailable" | `metrics.json` missing — run the notebook first |
| Frontend shows "API Offline" | API container still starting — wait a few seconds, check `docker-compose logs api` |
| Build fails on `COPY models/` | Run `mkdir models` (the directory must exist even if empty) |

## Tech Stack

| Category | Technology |
|----------|-----------|
| ML | XGBoost, Scikit-learn, LightGBM |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Data | Pandas, NumPy |
| Class Balancing | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow (optional) |

## License

MIT
