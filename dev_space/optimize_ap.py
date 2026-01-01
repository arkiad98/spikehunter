# dev_space/optimize_ap.py
# AP 개선을 위한 긴급 최적화 스크립트

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import joblib

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.utils_io import read_yaml
from modules.derive import _get_feature_cols

def load_data():
    """데이터 로드 및 전처리"""
    cfg = read_yaml(os.path.join(PROJECT_ROOT, "config/settings.yaml"))
    
    # 1. 학습 전용 데이터셋 우선 로드
    paths = cfg["paths"]
    dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    
    print(f">> Loading Data from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. 피처 선택
    feature_cols = _get_feature_cols(df.columns)
    
    # 3. 최근 36개월 데이터만 사용 (설정값 참조)
    train_months = cfg.get("ml_params", {}).get("classification_train_months", 36)
    cutoff_date = df['date'].max() - pd.DateOffset(months=train_months)
    df = df[df['date'] >= cutoff_date]
    
    print(f">> Data Shape: {df.shape}")
    print(f">> Target Distribution: {df['label_class'].value_counts(normalize=True)}")
    
    return df[feature_cols], df['label_class']

def objective(trial):
    X, y = load_data()
    
    # 하이퍼파라미터 탐색 공간
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbosity': -1,
        'n_jobs': 4,
        'random_state': 42,
        
        # 핵심 튜닝 대상
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000, step=100),
        
        # 보조 튜닝
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    
    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='average_precision',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds)
        scores.append(score)
        
    return np.mean(scores)

if __name__ == "__main__":
    print(">>> Starting AP Optimization (LightGBM)...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30) # 빠른 결과를 위해 30회 수행
    
    print("\n" + "="*60)
    print(f"BEST AP: {study.best_value:.4f}")
    print(f"BEST PARAMS: {study.best_params}")
    print("="*60)
    
    # 결과 파일 저장
    with open(os.path.join(PROJECT_ROOT, "dev_space", "ap_optimization_result.txt"), "w") as f:
        f.write(f"BEST AP: {study.best_value:.4f}\n")
        f.write(f"BEST PARAMS: {study.best_params}\n")
