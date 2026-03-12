# Multi-stage Dockerfile for Loan Approval Prediction System

# Base image with Python
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install only runtime dependencies (skip notebook-only packages)
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.3 \
    requests==2.31.0 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    joblib==1.3.2 \
    shap==0.44.1 \
    streamlit==1.29.0 \
    plotly==5.18.0

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create models directory (model.pkl is mounted at runtime via docker-compose)
RUN mkdir -p ./models

# ----------------- API Stage -----------------
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------- Frontend Stage -----------------
FROM base AS frontend

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
