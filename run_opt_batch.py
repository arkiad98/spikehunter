import sys
import os
from datetime import datetime

# Add project root to sys.path (though running from root makes this implicit, explicit is good)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.optimization import run_optimization_pipeline
from modules.utils_logger import setup_global_logger

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_global_logger(timestamp)
    
    print("Starting direct optimization pipeline...")
    # Using 'n_jobs=-1' (all cores), 'auto_approve=True'
    # warm_start=True is default but explicit is good
    run_optimization_pipeline("d:/spikehunter/config/settings.yaml", warm_start=True, n_jobs=-1, auto_approve=True)
    print("Optimization pipeline execution finished.")
