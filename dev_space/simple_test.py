# dev_space/simple_test.py
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from modules.utils_io import read_yaml
from modules.derive import _get_feature_cols

def run_test():
    print(">> Loading Data...")
    cfg = read_yaml(os.path.join(PROJECT_ROOT, "config/settings.yaml"))
    dataset_path = os.path.join(cfg["paths"]["features"], "dataset_v4.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(cfg["paths"]["ml_dataset"], "ml_classification_dataset.parquet")
    
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 6 months
    cutoff = df['date'].max() - pd.DateOffset(months=6)
    df = df[df['date'] >= cutoff].copy()
    
    # Clean cols
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    feature_cols = [c for c in df.columns if c not in ['date', 'label_class', 'label_regression']]
    
    X = df[feature_cols]
    y = df['label_class']
    
    print(f">> Data: {X.shape}, Pos Ratio: {y.mean():.4f}")
    
    # Test Config: High Weight
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbose': -1,
        'n_jobs': 4,
        'random_state': 42,
        'scale_pos_weight': 10.0, # Aggressive Weighting
        'learning_rate': 0.05,
        'n_estimators': 1000
    }
    
    print(f">> Training with scale_pos_weight={params['scale_pos_weight']}...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, preds)
    
    print(f"\n[RESULT] Val AP: {ap:.4f}")
    
    with open("dev_space/simple_result.txt", "w") as f:
        f.write(str(ap))

if __name__ == "__main__":
    run_test()
