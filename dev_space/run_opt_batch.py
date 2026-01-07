import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from modules.optimization import run_optimization_pipeline
from modules.utils_logger import setup_global_logger

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_global_logger(timestamp)
    # Direct call to optimization pipeline
    # This bypasses the menu in run_pipeline.py
    print("Starting direct optimization pipeline...")
    # Using 'clear_cache=False', 'n_jobs=-1' (all cores), 'auto_approve=True'
    run_optimization_pipeline(warm_start=True, n_jobs=-1, auto_approve=True)
    print("Optimization pipeline execution finished.")
