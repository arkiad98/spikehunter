import time
import os
import optuna
import pandas as pd
from unittest.mock import MagicMock
from modules import optimization

# Mock run_backtest to simulate work without side effects
def mock_run_backtest(*args, **kwargs):
    time.sleep(1.0) # Simulate 1 second of CPU work
    return {
        'metrics': {
            'Sharpe_raw': 1.0,
            'win_rate_raw': 0.5,
            'MDD_raw': -0.1,
            '총거래횟수': 10
        }
    }

optimization.run_backtest = mock_run_backtest

# Mock read_yaml to return dummy config
def mock_read_yaml(path):
    return {
        "paths": {"cache": "data/cache", "features": "data/proc/features", "models": "data/models"},
        "optimization": {
            "TestStrategy": {
                "param_space": {
                    "x": {"type": "float", "low": 0, "high": 10},
                    "y": {"type": "int", "low": 1, "high": 5}
                },
                "optimize_on": "Sharpe_raw"
            }
        },
        "strategies": {"TestStrategy": {"x": 1, "y": 1}}
    }
optimization.read_yaml = mock_read_yaml
optimization.logger = MagicMock()

def test_parallelism():
    print("Testing Parallel Optimization Logic with Joblib...")
    
    # Setup
    study = optuna.create_study(direction="maximize")
    df = pd.DataFrame({'close': [1, 2, 3]}) # Dummy DF
    n_trials = 8
    
    # Test 1: n_jobs = 1
    print("\n[Test 1] n_jobs=1 (Sequential)")
    start_time = time.time()
    optimization.run_batch_optimization(study, "dummy_path", "TestStrategy", "2020-01-01", "2020-01-02", df, n_trials=4, n_jobs=1)
    duration_seq = time.time() - start_time
    print(f"Duration: {duration_seq:.2f}s (Expected ~4.0s)")
    
    # Test 2: n_jobs = 4
    print("\n[Test 2] n_jobs=4 (Parallel)")
    study = optuna.create_study(direction="maximize") # Reset study
    start_time = time.time()
    optimization.run_batch_optimization(study, "dummy_path", "TestStrategy", "2020-01-01", "2020-01-02", df, n_trials=4, n_jobs=4)
    duration_par = time.time() - start_time
    print(f"Duration: {duration_par:.2f}s (Expected ~1.0s + overhead)")
    
    # Verification
    if duration_par < duration_seq * 0.5:
        print("\nSUCCESS: Parallel execution is significantly faster.")
    else:
        print("\nWARNING: Parallel execution did not show expected speedup. (Are we on a 1-core machine?)")

if __name__ == "__main__":
    test_parallelism()
