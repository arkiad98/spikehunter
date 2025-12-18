import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from modules.utils_io import read_yaml, optimize_memory_usage
from modules.derive import _get_feature_cols
from modules.train import _get_core_features_from_registry

def optimize_threshold(settings_path="config/settings.yaml"):
    cfg = read_yaml(settings_path)
    
    # 1. Load Model
    model_path = os.path.join(cfg['paths']['models'], "lgbm_model.joblib")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f">> Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Load Data
    dataset_path = os.path.join(cfg['paths']['ml_dataset'], "ml_classification_dataset.parquet")
    print(f">> Loading data from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df = optimize_memory_usage(df)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Use the last 6 months for validation (Test Set)
    test_months = 6
    cutoff_date = df['date'].max() - pd.DateOffset(months=test_months)
    test_df = df[df['date'] >= cutoff_date].copy()
    
    print(f">> Test Data: {len(test_df)} rows (since {cutoff_date.date()})")
    
    # 3. Features
    registry_path = "config/feature_registry.yaml"
    core_features = _get_core_features_from_registry(registry_path)
    available_cols = set(test_df.columns)
    feature_cols = [f for f in core_features if f in available_cols]
    if not feature_cols: feature_cols = _get_feature_cols(test_df.columns)
    
    X_test = test_df[feature_cols]
    y_test = test_df['label_class']
    
    # 4. Predict
    print(">> Predicting probabilities...")
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test) # Fallback if predict returns probs (unlikely for sklearn API)
        
    # 5. Threshold Analysis
    print("\n[Threshold Optimization Analysis]")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Pred Count':<10}")
    print("-" * 60)
    
    best_f1 = 0
    best_th = 0.5
    
    for th in np.arange(0.1, 0.95, 0.05):
        preds = (probs >= th).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        count = preds.sum()
        
        print(f"{th:.2f}       | {prec:.4f}     | {rec:.4f}     | {f1:.4f}     | {count}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    print("-" * 60)
    print(f"Best Threshold (F1): {best_th:.2f} (F1: {best_f1:.4f})")
    
    ap = average_precision_score(y_test, probs)
    print(f"Average Precision (AP): {ap:.4f}")

if __name__ == "__main__":
    optimize_threshold()
