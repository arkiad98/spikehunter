import os
import datetime
import pandas as pd
from modules.train import run_train_pipeline
from modules.backtest import run_backtest

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

def run_experiment():
    print(">>> Starting Experiment: Feature Optimization")
    
    # 2. Train
    print("\n>>> Step 2: Training Model")
    try:
        run_train_pipeline("config/settings.yaml")
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # 3. Backtest
    print("\n>>> Step 3: Backtesting")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/proc/backtest/exp_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        run_backtest(
            run_dir=run_dir,
            strategy_name="Feature_Opt_Experiment",
            settings_path="config/settings.yaml"
        )
    except Exception as e:
        print(f"Backtest failed: {e}")
        return

    print(f"\n>>> Experiment Complete. Results saved in {run_dir}")

if __name__ == "__main__":
    run_experiment()
