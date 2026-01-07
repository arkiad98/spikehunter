import os
import sys
import pandas as pd
from datetime import datetime
from ruamel.yaml import YAML

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.utils_io import read_yaml
from modules.utils_logger import logger
from modules.backtest import run_backtest
from modules.optimization import objective  # If possible, or just replicate the call

def debug_optimization_failure():
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    strategy_name = "SpikeHunter_R1_BullStable"
    
    # Failing Parameters from Trial 0 (Warm Start)
    params = {
        'target_r': 0.09990738335325917, 
        'stop_r': -0.027337402392375594, 
        'min_ml_score': 0.21000000000000005, 
        'min_ml_score': 0.21000000000000005, 
        'max_market_vol': 0.015, # [Debug] Stricter Volatility to verify filter
        'vbo_k': 0.5417664494167612, 
        'min_mfi': 46.79727604441189
    }
    
    print(">>> Debugging Optimization Failure for Trial 0 Params")
    print(f"Params: {params}")
    print(f"Testing with Stricter Volatility: {params['max_market_vol']}")

    # 1. Load Data like optimization.py does
    paths = cfg["paths"]
    dataset_path = os.path.join(paths.get("ml_dataset", "data/proc/ml_dataset"), "ml_classification_dataset.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
        
    print(f"Loading Dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Columns: {df.columns.tolist()}") # [Debug] Check columns
    
    # Pre-calc ML Score (Simulate Optimization behavior)
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)
        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is not None:
            missing = [c for c in feature_names if c not in df.columns]
            for c in missing: df[c] = 0
            X_temp = df[feature_names].fillna(0)
            scores = model.predict_proba(X_temp)[:, 1]
            df['ml_score'] = scores
            print(f"ML Score Pre-calculated. Mean: {scores.mean():.4f}, Max: {scores.max():.4f}")
    
    # 2. Run Backtest
    start_date = "2020-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    temp_run_dir = os.path.join(paths["cache"], "debug_opt_trial_0")
    
    print(f"Running Backtest ({start_date} ~ {end_date})...")
    
    try:
        result = run_backtest(
            run_dir=temp_run_dir,
            strategy_name=strategy_name,
            settings_cfg=cfg,
            start=start_date,
            end=end_date,
            param_overrides=params,
            quiet=False,
            preloaded_features=df,
            save_to_db=False,
            skip_exclusion=True # Optimization does this
        )
        
        if not result:
            print(">>> Result is None (Failure in run_backtest)")
        else:
            metrics = result['metrics']
            print("\n>>> Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            
            # Check constraints
            win_rate = metrics.get('win_rate_raw', 0.0)
            mdd = metrics.get('MDD_raw', -1.0)
            trades = metrics.get('총거래횟수', 0)
            
            print(f"\n>>> Checks:")
            print(f"  Win Rate ({win_rate:.4f}) < 0.2? {win_rate < 0.2}")
            print(f"  MDD ({mdd:.4f}) < -0.30? {mdd < -0.30}")
            print(f"  Trades ({trades}) < 3? {trades < 3}")
            
    except Exception as e:
        print(f">>> Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_optimization_failure()
