import sys
import os
import optuna
import joblib
import pandas as pd
import numpy as np
from modules.utils_io import read_yaml, update_yaml
from modules.utils_logger import logger
from modules.train import Objective, _get_core_features_from_registry, _get_feature_cols

# Mock input to avoid interaction if needed, but we will call logic directly
def run_auto_optimization():
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    
    # 1. Optimize LightGBM
    print("\n[Auto] Starting LightGBM Optimization...")
    optimize_model(cfg, settings_path, model_type='lgbm', target_key='lgbm_params_classification', space_key='param_space_lgbm')
    
    # Reload config to get updated params if any (though we update file, cfg variable is old)
    cfg = read_yaml(settings_path)

    # 2. Optimize XGBoost
    print("\n[Auto] Starting XGBoost Optimization...")
    optimize_model(cfg, settings_path, model_type='xgb', target_key='xgb_params_classification', space_key='param_space_xgb')

    # Reload config
    cfg = read_yaml(settings_path)

    # 3. Optimize CatBoost
    print("\n[Auto] Starting CatBoost Optimization...")
    optimize_model(cfg, settings_path, model_type='cat', target_key='cat_params_classification', space_key='param_space_cat')

def optimize_model(cfg, settings_path, model_type, target_key, space_key):
    # Fix: Load param_space from ml_params -> target_key -> space_key
    ml_params = cfg.get("ml_params", {})
    model_params = ml_params.get(target_key, {})
    
    if space_key in model_params:
        param_space = model_params[space_key]
    else:
        # Fallback to old location if not found (though unlikely now)
        ml_opt_cfg = cfg.get("ml_optimization", {}).get("classification", {})
        param_space = ml_opt_cfg.get(space_key, {})
        
    # Set n_trials to 100 for deep optimization
    n_trials = 100 
    
    # [User Request] Active Multiprocessing
    # Optuna n_jobs=4 (4 concurrent trials)
    # Model n_jobs=2 (2 threads per model) -> Total ~8 threads active
    optuna_jobs = 4
    model_jobs = 2
    
    print(f" >> Model: {model_type}, Trials: {n_trials}")
    print(f" >> Parallelism: {optuna_jobs} Concurrent Trials x {model_jobs} Model Threads")
    print(f" >> Loaded Param Space Keys: {list(param_space.keys())}")
    
    study = optuna.create_study(direction="maximize")
    objective = Objective(model_type, param_space, "AP", model_jobs)
    
    try:
        # Run optimization with parallel trials
        study.optimize(objective, n_trials=n_trials, n_jobs=optuna_jobs, show_progress_bar=True)
        
        print(f"\n [Optimization Complete] Best AP: {study.best_value:.4f}")
        print(f" Best Params: {study.best_params}")
        
        # Save to settings.yaml
        base_params = cfg['ml_params'].get(target_key, {})
        base_params.update(study.best_params)
        update_yaml(settings_path, "ml_params", target_key, base_params)
        print(f" >> Saved optimized parameters to {target_key}")
            
    except Exception as e:
        logger.error(f"Optimization failed for {model_type}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_auto_optimization()
