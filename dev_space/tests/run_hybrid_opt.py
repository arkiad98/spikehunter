
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath("d:/spikehunter"))

from modules.utils_logger import logger
from modules.optimization import optimize_strategy_headless

def run():
    logger.info("Starting Hybrid Strategy Optimization...")
    best_params = optimize_strategy_headless(
        settings_path="d:/spikehunter/config/settings.yaml",
        strategy_name="SpikeHunter_R1_BullStable",
        start_date="2020-01-01",
        end_date="2024-12-31",
        n_trials=30,  # Quick validation
        n_jobs=-1
    )
    
    print("\n" + "="*50)
    print("Optimization Result (Hybrid Stability):")
    print(best_params)
    print("="*50)

if __name__ == "__main__":
    run()
