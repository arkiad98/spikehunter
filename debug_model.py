
import os
import pandas as pd
import joblib
import numpy as np
import lightgbm as lgb
from modules.utils_io import read_yaml

def debug_model():
    cfg = read_yaml("config/settings.yaml")
    paths = cfg["paths"]
    
    # 1. Load Model
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    print(f"Model Type: {type(model)}")
    
    # Check expected features
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        print(f"\nModel Expects {len(expected_features)} Features: {expected_features}")
    else:
        print("Model does not have feature_names_in_ attribute.")
        expected_features = []

    # 2. Load Data
    dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    print(f"\nDataset Info:")
    print(f"Latest Date: {latest_date}")
    print(f"Total Rows: {len(df)}")
    
    # 3. Analyze Latest Day Data
    df_latest = df[df['date'] == latest_date].copy()
    print(f"Latest Day Rows: {len(df_latest)}")
    
    # Check for missing columns
    missing_cols = [c for c in expected_features if c not in df_latest.columns]
    if missing_cols:
        print(f"CRITICAL: Missing columns in dataset: {missing_cols}")
    else:
        print("All expected features are present in the dataset.")
        
    # 4. Check Feature Statistics
    print("\nFeature Statistics (Latest Day):")
    X = df_latest[expected_features]
    stats = X.describe().T[['mean', 'std', 'min', 'max']]
    print(stats)
    
    # 5. Run Prediction
    print("\nRunning Prediction...")
    try:
        preds = model.predict_proba(X)[:, 1]
        df_latest['ml_score'] = preds
        
        print(f"Score Stats:\n{df_latest['ml_score'].describe()}")
        
        high_score_count = (preds >= 0.4).sum()
        print(f"\nStocks with Score >= 0.4: {high_score_count} / {len(df_latest)}")
        
        high_score_samples = df_latest[df_latest['ml_score'] >= 0.4].head(5)
        print("\nSample High Score Stocks:")
        print(high_score_samples[['code', 'close', 'ml_score'] + list(expected_features[:5])])
        
    except Exception as e:
        print(f"Prediction Error: {e}")

if __name__ == "__main__":
    debug_model()
