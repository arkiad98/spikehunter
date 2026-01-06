
import sys
import os
import pandas as pd
import logging

# Setup path
sys.path.append(os.path.abspath("D:/spikehunter"))

from modules.backtest import run_backtest
from modules.utils_logger import logger

# Configure logger to print to stdout
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Params from Trial 8 (Low score, likely to generate trades)
# Trial 8 finished with value: -999.0 and parameters: 
# {'target_r': 0.0634, 'stop_r': -0.0495, 'min_ml_score': 0.2349, 'max_market_vol': 0.029, 'vbo_k': 0.526, 'min_mfi': 47.6}

params = {
    'target_r': 0.0634, 
    'stop_r': -0.0495, 
    'min_ml_score': 0.2349, 
    'max_market_vol': 0.029, 
    'vbo_k': 0.526, 
    'min_mfi': 47.6
}

print(f"Running Debug Backtest with params: {params}")

try:
    result = run_backtest(
        run_dir="D:/spikehunter/data/cache/debug_run",
        strategy_name="SpikeHunter_R1_BullStable",
        settings_path="D:/spikehunter/config/settings.yaml",
        start="2020-01-01",
        end="2026-01-02",
        param_overrides=params,
        quiet=False,
        save_to_db=False,
        skip_exclusion=True
    )

    if result:
        metrics = result['metrics']
        print("\n--- Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        # Check optimization constraints
        win_rate = metrics.get('win_rate_raw', 0)
        mdd = metrics.get('MDD_raw', 0)
        
        print("\n--- Constraints Check ---")
        print(f"Win Rate < 0.3? {win_rate < 0.3} ({win_rate:.4f})")
        print(f"MDD < -0.15? {mdd < -0.15} ({mdd:.4f})")
        
        if win_rate < 0.3 or mdd < -0.15:
            print("Verdict: REJECTED")
        else:
            print("Verdict: ACCEPTED")
    else:
        print("Result is None")

except Exception as e:
    import traceback
    traceback.print_exc()
