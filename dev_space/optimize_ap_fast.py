# dev_space/optimize_ap_fast.py
# AP 개선을 위한 고속 최적화 (데이터 샘플링 적용)

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.utils_io import read_yaml
from modules.derive import _get_feature_cols

def load_data_sample():
    """최근 데이터 중심 샘플링"""
    cfg = read_yaml(os.path.join(PROJECT_ROOT, "config/settings.yaml"))
    paths = cfg["paths"]
    
    dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    
    print(f">> Loading Data from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # [Fast Mode] 최근 6개월 데이터만 사용 (안정성 확보)
    cutoff_date = df['date'].max() - pd.DateOffset(months=6)
    df = df[df['date'] >= cutoff_date]
    
    # [Fix] LightGBM Column Name Issue (JSON characters)
    import re
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    feature_cols = [c for c in df.columns if c not in ['date', 'label_class', 'label_regression']]
    
    print(f">> Sampled Data Shape: {df.shape}")
    print(f">> Target Distribution: {df['label_class'].value_counts(normalize=True)}")
    
    return df[feature_cols], df['label_class']

def objective(trial):
    X, y = load_data_sample()
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbosity': -1,
        'n_jobs': 4,
        'random_state': 42,
        
        # 탐색 범위 확장 (과감한 가중치)
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, step=200),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
    }
    
    # K-Fold 줄임 (속도 우선)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds)
        scores.append(score)
        
    return np.mean(scores)

if __name__ == "__main__":
    print(">>> Starting Fast AP Optimization...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50) # 50회 시도
    
    print("\n" + "="*60)
    print(f"BEST AP: {study.best_value:.4f}")
    print(f"BEST PARAMS: {study.best_params}")
    print("="*60)
    
    with open(os.path.join(PROJECT_ROOT, "dev_space", "ap_optimization_result_fast.txt"), "w") as f:
        f.write(f"BEST AP: {study.best_value:.4f}\n")
        f.write(f"BEST PARAMS: {study.best_params}\n")
