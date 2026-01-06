
import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from modules.utils_logger import logger # Import logger to match project style if needed

def analyze_distribution():
    print("="*60)
    print("   ML Score Distribution Analysis (Probability Mode)")
    print("="*60)
    
    # Paths
    dataset_path = "d:/spikehunter/data/proc/ml_dataset/ml_classification_dataset.parquet"
    model_path = "d:/spikehunter/data/models/lgbm_model.joblib"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load Data
    print(f"Loading dataset: {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load Model
    print(f"Loading model: {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare Features
    try:
         # Try to get feature names from model
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
        elif hasattr(model, 'feature_name'): # Booster
             features = model.feature_name()
        else:
             print("Could not determine feature names from model.")
             # Fallback: assume all columns except date/label/code are features or user explicit list
             # For now, let's just try to predict on common numerical columns if fail
             features = df.select_dtypes(include=[np.number]).columns.tolist()
             if 'label' in features: features.remove('label')
             
        # Check missing features
        missing = [c for c in features if c not in df.columns]
        if missing:
            print(f"CRITICAL: Missing features in dataset: {missing}")
            # Add dummy 0
            for c in missing: df[c] = 0
            
        X = df[features].fillna(0)
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return

    # Predict Probabilities
    print("Predicting probabilities...")
    try:
        if hasattr(model, "predict_proba"):
            # Sklearn API
            preds = model.predict_proba(X)[:, 1]
            mode = "Sklearn (predict_proba)"
        else:
            # Booster API
            preds = model.predict(X)
            mode = "Booster (predict)"
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Analyze Stats
    print(f"\n[Prediction Mode: {mode}]")
    print(f"Min: {preds.min():.4f}")
    print(f"Max: {preds.max():.4f}")
    print(f"Mean: {preds.mean():.4f}")
    print(f"Median: {np.median(preds):.4f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(preds, percentiles)
    print("\n[Percentiles]")
    for p, v in zip(percentiles, perc_vals):
        print(f" {p}%: {v:.4f}")
        
    # Check Threshold Candidates
    print("\n[Threshold Candidates Count in recent 6 months (Test Period)]")
    test_df = df[df['date'] >= '2025-06-30'].copy()
    test_rows = len(test_df)
    
    if test_rows == 0:
        print("No test data found > 2025-06-30")
    else:
        test_X = test_df[features].fillna(0)
        if hasattr(model, "predict_proba"):
            test_preds = model.predict_proba(test_X)[:, 1]
        else:
            test_preds = model.predict(test_X)
            
        thresholds = [0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.50]
        for th in thresholds:
            count = np.sum(test_preds >= th)
            ratio = count / test_rows * 100
            print(f" Thresh {th:.2f}: {count} trades ({ratio:.2f}%)")

if __name__ == "__main__":
    analyze_distribution()
