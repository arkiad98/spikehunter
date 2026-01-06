
import os
import glob

def clean_temp_files():
    files = ['debug_trial.py', 'debug_dataset_cols.py', 'analyze_volatility_trend.py']
    for f in files:
        if os.path.exists(f):
            print(f"Removing {f}")
            os.remove(f)

if __name__ == "__main__":
    clean_temp_files()
