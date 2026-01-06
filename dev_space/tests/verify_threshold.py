import sys
import os
# Add project root to path
sys.path.append(os.getcwd())

from modules.optimization import optimize_strategy_headless
from modules.utils_logger import logger

# Setup
settings_path = 'config/settings.yaml'
strategy_name = 'SpikeHunter_R1_BullStable'
start_date = '2024-01-01'
end_date = '2024-02-01'

print(f"Starting Threshold Verification for {strategy_name} ({start_date} ~ {end_date})")

try:
    best_params = optimize_strategy_headless(
        settings_path=settings_path,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        n_trials=1,  # Ultra Fast check
        n_jobs=1
    )
    
    print("\n[Verification Result]")
    print(f"Best Params: {best_params}")
    
    if best_params and 'min_ml_score' in best_params:
        print(f"SUCCESS: Threshold (min_ml_score) optimized: {best_params['min_ml_score']}")
    else:
        print("FAILURE: min_ml_score not found in best params or optimization failed.")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
