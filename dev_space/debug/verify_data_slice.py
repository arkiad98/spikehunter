
from modules.optimization_ml import load_data_cached
from modules.utils_logger import setup_global_logger
import pandas as pd

if __name__ == "__main__":
    setup_global_logger("debug_slice")
    print(">> Verifying load_data_cached with offset...")
    
    try:
        X, y = load_data_cached("config/settings.yaml")
        
        if X is not None:
             print(f"Slice Success! X.shape: {X.shape}, y.shape: {y.shape}")
             # Check integrity
             print(f"X NaNs: {X.isna().sum().sum()}")
             print(f"y NaNs: {y.isna().sum()}")
        else:
             print("Slice Failed! X is None.")
             
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
