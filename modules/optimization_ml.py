# modules/optimization_ml.py
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import re
import joblib
from functools import lru_cache
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_score

from modules.utils_io import read_yaml, update_yaml, get_user_input
from modules.utils_logger import logger

# Global Data Cache
_DATA_CACHE: Dict[str, Any] = None

def load_data_cached(settings_path: str, months_back: int = 18) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and cache ML dataset with optimization-friendly preprocessing.
    """
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        logger.info("[ML-Opt] Using cached dataset.")
        return _DATA_CACHE['X'], _DATA_CACHE['y']
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # Try multiple paths for robustness
    dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
        
    if not os.path.exists(dataset_path):
        logger.error(f"[ML-Opt] Dataset not found at {dataset_path}")
        return None, None
        
    logger.info(f"[ML-Opt] Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Recent data filter
    cutoff_date = df['date'].max() - pd.DateOffset(months=months_back)
    df = df[df['date'] >= cutoff_date].copy()
    
    # Sanitize Columns for LightGBM
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # Feature Selection
    exclude_cols = ['date', 'code', 'name', 'label_class', 'label_regression', 'change_rate', 'close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols]
    y = df['label_class']
    
    # Update Cache
    _DATA_CACHE = {'X': X, 'y': y}
    
    logger.info(f"[ML-Opt] Data Loaded. Shape: {X.shape}, Pos Ratio: {y.mean():.4f}")
    return X, y

def calculate_top_n_precision(y_true, y_scores, n=5):
    """
    Calculate Precision of Top-N items per day.
    (Approximation using overall top-N for simplicity in optimization loop, 
    or just top-k percentile since precise daily grouping is slow in loop)
    
    For optimization speed, we use top-k percentile as proxy for top-n strategy.
    """
    if len(y_true) == 0: return 0.0
    
    # Simply take top 5% as proxy for "Trade Candidates"
    # Or strict top k items
    k = max(int(len(y_true) * 0.05), 5) 
    
    # Get indices of top k scores
    try:
        top_k_indices = np.argsort(y_scores)[-k:]
        y_true_top_k = y_true.iloc[top_k_indices] if hasattr(y_true, 'iloc') else y_true[top_k_indices]
        return np.mean(y_true_top_k)
    except:
        return 0.0

def objective(trial, settings_path: str):
    X, y = load_data_cached(settings_path)
    if X is None: return -1.0
    
    # Load Search Space from settings.yaml
    cfg = read_yaml(settings_path)
    lgbm_cfg = cfg['ml_params']['lgbm_params_classification']
    param_space = lgbm_cfg.get('param_space_lgbm', {})
    
    # Base Params
    param = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbosity': -1,
        'n_jobs': 4,
        'boosting_type': 'gbdt',
        'random_state': 42
    }
    
    # Suggest Params dynamic logic
    for key, config in param_space.items():
        if config['type'] == 'int':
            param[key] = trial.suggest_int(key, config['low'], config['high'])
        elif config['type'] == 'float':
            log_scale = config.get('log', False)
            param[key] = trial.suggest_float(key, config['low'], config['high'], log=log_scale)
        elif config['type'] == 'categorical':
            param[key] = trial.suggest_categorical(key, config['choices'])
            
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    scores_ap = []
    scores_top_prec = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            param, dtrain, 
            valid_sets=[dval],
            callbacks=[
                # Quiet mode
                lgb.log_evaluation(period=0)
            ]
        )
        
        preds = model.predict(X_val)
        ap = average_precision_score(y_val, preds)
        
        # [New] Auxiliary Metric: Top-N Precision
        top_prec = calculate_top_n_precision(y_val, preds)
        
        scores_ap.append(ap)
        scores_top_prec.append(top_prec)
        
    # Set user attribute for analysis
    mean_ap = np.mean(scores_ap)
    mean_top_prec = np.mean(scores_top_prec)
    
    trial.set_user_attr("top_n_precision", mean_top_prec)
    
    return mean_ap

def analyze_importance(study):
    """
    Analyze and print hyperparameter importances.
    """
    try:
        logger.info("\n" + "="*40)
        logger.info("   [Hyperparameter Importance Analysis]")
        logger.info("="*40)
        
        importance = optuna.importance.get_param_importances(study)
        
        for i, (param, score) in enumerate(importance.items()):
            logger.info(f" {i+1}. {param:<20}: {score*100:.1f}%")
            
        logger.info("="*40 + "\n")
            
    except Exception as e:
        logger.warning(f"Importance analysis failed: {e}")

def run_ml_optimization(settings_path: str, n_trials: int = 20):
    logger.info("="*60)
    logger.info("      <<< ML Model Hyperparameter Optimization >>>")
    logger.info("      * Analyzing Scale Pos Weight, Tree Depth, etc.")
    logger.info("="*60)
    
    # Create Study
    study = optuna.create_study(direction="maximize")
    
    logger.info(f">> Running {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, settings_path), n_trials=n_trials)
    
    logger.info("\n" + "="*60)
    logger.info(f" [Optimization Result]")
    logger.info(f" Best AP: {study.best_value:.4f}")
    
    best_trial = study.best_trial
    top_n_prec = best_trial.user_attrs.get("top_n_precision", 0.0)
    logger.info(f" Top-N Precision (Proxy): {top_n_prec:.4f}")
    logger.info(f" Best Params: {study.best_params}")
    logger.info("="*60)
    
    # Analyze Importance
    analyze_importance(study)
    
    # Save/Update Question
    q = get_user_input("Update settings.yaml with these parameters? (y/n): ")
    if q.lower() == 'y':
        update_yaml(settings_path, "ml_params", "lgbm_params_classification", study.best_params)
        logger.info(">> settings.yaml updated successfully.")
        
if __name__ == "__main__":
    # Test Run
    run_ml_optimization("config/settings.yaml", n_trials=5)
