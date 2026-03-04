"""
Test script to verify all components
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import mlflow
        import fastapi
        import streamlit
        import plotly
        import requests
        print("✅ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    base_dir = Path(__file__).parent
    
    required_paths = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "backend",
        "backend/data_pipeline",
        "backend/model",
        "frontend",
    ]
    
    all_exist = True
    for path in required_paths:
        full_path = base_dir / path
        if full_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_config_files():
    """Test if configuration files exist"""
    print("\nTesting configuration files...")
    base_dir = Path(__file__).parent
    
    config_files = [
        "requirements.txt",
        "params.yaml",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
    ]
    
    all_exist = True
    for file in config_files:
        full_path = base_dir / file
        if full_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_python_files():
    """Test if all Python files exist"""
    print("\nTesting Python files...")
    base_dir = Path(__file__).parent
    
    python_files = [
        "backend/data_pipeline/preprocess.py",
        "backend/data_pipeline/feature_engineering.py",
        "backend/model/train.py",
        "backend/model/predict.py",
        "backend/main.py",
        "frontend/streamlit_app.py",
    ]
    
    all_exist = True
    for file in python_files:
        full_path = base_dir / file
        if full_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_dataset():
    """Test if dataset is present"""
    print("\nTesting dataset...")
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "data" / "raw" / "cs-training.csv"
    
    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path}")
        import pandas as pd
        df = pd.read_csv(dataset_path, nrows=5)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return True
    else:
        print(f"⚠️  Dataset NOT found: {dataset_path}")
        print("   Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data")
        return False

def test_model():
    """Test if trained model exists"""
    print("\nTesting model...")
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "model.pkl"
    
    if model_path.exists():
        print(f"✅ Model found: {model_path}")
        try:
            import joblib
            model = joblib.load(model_path)
            print(f"   Model type: {type(model).__name__}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"⚠️  Model NOT found: {model_path}")
        print("   Run: python backend/model/train.py")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("  LOAN DEFAULT PREDICTION - SYSTEM TEST")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Project Structure": test_project_structure(),
        "Config Files": test_config_files(),
        "Python Files": test_python_files(),
        "Dataset": test_dataset(),
        "Model": test_model()
    }
    
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Run: python backend/model/train.py  (if model not found)")
        print("  2. Run: start_services.bat")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing dataset: Download from Kaggle")
        print("  - Missing model: python backend/model/train.py")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
