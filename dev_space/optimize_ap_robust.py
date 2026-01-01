# dev_space/optimize_ap_robust.py
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import re
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.utils_io import read_yaml
from modules.derive import _get_feature_cols

def load_data_safe():
    cfg = read_yaml(os.path.join(PROJECT_ROOT, "config/settings.yaml"))
    
    # Fixed path access
    try:
        dataset_path = os.path.join(cfg["paths"]["ml_dataset"], "ml_classification_dataset.parquet")
    except KeyError:
        dataset_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
        
    print(f">> Loading: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Use recent 18 months for optimization balance
    cutoff_date = df['date'].max() - pd.DateOffset(months=18)
    df = df[df['date'] >= cutoff_date].copy()
    
    # [Fix] Sanitize Column Names for LightGBM
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # Identify features
    exclude_cols = ['date', 'code', 'name', 'label_class', 'label_regression', 'change_rate', 'close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Filter numeric only just in case
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    print(f">> Data Shape: {df.shape}")
    print(f">> Feature Count: {len(feature_cols)}")
    print(f">> Positive Ratio: {df['label_class'].mean():.4f}")
    
    return df[feature_cols], df['label_class']

def objective(trial):
    X, y = load_data_safe()
    
    param = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbosity': -1,
        'n_jobs': 4,
        'boosting_type': 'gbdt',
        'random_state': 42,
        
        # Hyperparams
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    
    # TimeSeriesSplit (3 folds for speed in optimization)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Use simple train to avoid overhead
        model = lgb.train(
            param, dtrain, 
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        
        preds = model.predict(X_val)
        score = average_precision_score(y_val, preds)
        scores.append(score)
        
    return np.mean(scores)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print(">> Starting Optimization (20 trials)...")
    study.optimize(objective, n_trials=20) # Fast check
    
    print("\n[Result]")
    print(f"Best AP: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    
    # Save to file
    with open("dev_space/optimized_params.txt", "w") as f:
        f.write(str(study.best_params))
