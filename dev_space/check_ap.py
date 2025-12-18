import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from modules.train import _run_classification_training
from modules.utils_io import read_yaml

def main():
    try:
        cfg = read_yaml("config/settings.yaml")
        # Force verbose to 0 to reduce noise if possible, though function prints anyway
        # We just want the final score
        _, ap_score = _run_classification_training(cfg, return_results_only=True)
        print(f"CURRENT_AP_SCORE:{ap_score}")
    except Exception as e:
        print(f"ERROR:{e}")

if __name__ == "__main__":
    main()
