
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath("d:/spikehunter"))

from modules.utils_logger import logger
from modules.optimization import optimize_strategy_headless

def run():
    logger.info("Starting Global Strategy Optimization (Hybrid Stability)...")
    
    # 1. Full Dataset Path
    dataset_path = "d:/spikehunter/data/proc/ml_dataset/ml_classification_dataset.parquet"
    
    # 2. Run Optimization
    best_params = optimize_strategy_headless(
        settings_path="d:/spikehunter/config/settings.yaml",
        strategy_name="SpikeHunter_R1_BullStable",
        start_date="2020-01-01",
        end_date="2025-06-30", # Full Range
        n_trials=100,  # Deep search
        n_jobs=-1,
        dataset_path=dataset_path # Explicit Override
    )
    
    print("\n" + "="*50)
    print("Global Optimization Result (Hybrid Stability):")
    print(best_params)
    print("="*50)

if __name__ == "__main__":
    run()
