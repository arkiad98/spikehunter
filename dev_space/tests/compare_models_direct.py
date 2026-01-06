
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from modules.utils_io import read_yaml
from modules.derive import _get_feature_cols
import logging

# Setup simple logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Compare")

def load_data(settings_path):
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # Try multiple paths (WF compatible)
    train_split_path = os.path.join(paths["features"], "dataset_v4.parquet")
    if os.path.exists(train_split_path):
        dataset_path = train_split_path
        logger.info(f"Using Split Dataset: {dataset_path}")
    else:
        dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
        logger.info(f"Using Full Dataset: {dataset_path}")
        
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter numeric features
    exclude_cols = ['date', 'code', 'name', 'label_class', 'label_regression', 'change_rate', 'close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    ml_params = cfg.get("ml_params", {})
    train_months = ml_params.get("classification_train_months", 36)
    cutoff_date = df['date'].max() - pd.DateOffset(months=train_months)
    df = df[df['date'] >= cutoff_date].copy()
    
    logger.info(f"Data Shape: {df.shape}, Features: {len(feature_cols)}")
    return df, feature_cols

def evaluate_params(name, params, X, y, splits=5):
    logger.info(f"\n>> Evaluating: {name}")
    tscv = TimeSeriesSplit(n_splits=splits)
    
    metrics = {
        'AP': [], 'AUC': [], 'F1': [], 'Precision': [], 'Recall': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        p = params.copy()
        p['verbose'] = -1
        p['objective'] = 'binary'
        p['metric'] = 'average_precision'
        p['n_jobs'] = 4
        
        model = lgb.LGBMClassifier(**p)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int) # Fixed threshold for comparison (or use smart one?)
        # Let's use 0.5 for F1/Prec/Recall foundation, but AP/AUC are threshold agnostic
        
        metrics['AP'].append(average_precision_score(y_val, probs))
        metrics['AUC'].append(roc_auc_score(y_val, probs))
        metrics['F1'].append(f1_score(y_val, preds, zero_division=0))
        metrics['Precision'].append(precision_score(y_val, preds, zero_division=0))
        metrics['Recall'].append(recall_score(y_val, preds, zero_division=0))
        
        # logger.info(f"  Fold {fold+1}: AP={metrics['AP'][-1]:.4f}")

    results = {k: np.mean(v) for k, v in metrics.items()}
    logger.info(f"  [Result] AP: {results['AP']:.4f} | AUC: {results['AUC']:.4f}")
    return results

def main():
    settings_path = "config/settings.yaml"
    df, features = load_data(settings_path)
    X = df[features]
    y = df['label_class']
    
    # 1. Baseline (Current Settings)
    params_baseline = {
        'scale_pos_weight': 8.304453082647894,
        'n_estimators': 540,
        'learning_rate': 0.012763748069217377,
        'num_leaves': 94,
        'max_depth': 17,
        'colsample_bytree': 0.9125967578720843,
        'subsample': 0.8480084483844194,
        'min_child_samples': 99
    }
    
    # 2. Trial 57 (New Candidate)
    params_trial57 = {
        'learning_rate': 0.005710216158576237, 
        'num_leaves': 150, 
        'max_depth': 5, 
        'n_estimators': 591, 
        'scale_pos_weight': 8.942341498709272, 
        'colsample_bytree': 0.8526828452795893, 
        'subsample': 0.7868119721463169, 
        'min_child_samples': 71
    }
    
    res_base = evaluate_params("Baseline (Current)", params_baseline, X, y)
    res_new = evaluate_params("Trial 57 (New)", params_trial57, X, y)
    
    print("\n" + "="*50)
    print(f"{'Metric':<10} | {'Baseline':<10} | {'Trial 57':<10} | {'Diff'}")
    print("-" * 50)
    for m in ['AP', 'AUC', 'F1', 'Precision', 'Recall']:
        diff = res_new[m] - res_base[m]
        mark = "▲" if diff > 0 else "▼"
        print(f"{m:<10} | {res_base[m]:.4f}     | {res_new[m]:.4f}     | {mark} {diff:+.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
