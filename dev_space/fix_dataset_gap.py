
import pandas as pd
import os
from modules.utils_logger import logger 

def merge_datasets():
    print("Loading datasets...")
    # 1. Load Legacy Feature Data (Has 2025 Sep-Dec)
    path_legacy = "data/proc/features/dataset_v4.parquet.bak"
    if not os.path.exists(path_legacy):
        print(f"Error: {path_legacy} not found")
        return

    df_legacy = pd.read_parquet(path_legacy)
    print(f"Legacy Loaded: {len(df_legacy)} rows, Max Date: {df_legacy['date'].max()}")

    # 2. Load New Feature Data (Has 2026 Jan, but gap)
    path_new = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    if not os.path.exists(path_new):
        print(f"Error: {path_new} not found")
        return
        
    df_new = pd.read_parquet(path_new)
    print(f"New Loaded: {len(df_new)} rows, Max Date: {df_new['date'].max()}")

    # 3. Filter New Data to only keep 2026+ (or anything after legacy)
    legacy_max_date = pd.to_datetime(df_legacy['date'].max())
    df_new['date'] = pd.to_datetime(df_new['date'])
    
    df_new_filtered = df_new[df_new['date'] > legacy_max_date].copy()
    print(f"New Data (Filtered > {legacy_max_date.date()}): {len(df_new_filtered)} rows")

    # 4. Concatenate
    df_final = pd.concat([df_legacy, df_new_filtered], ignore_index=True)
    df_final = df_final.sort_values(['date', 'code']).reset_index(drop=True)
    
    print(f"Final Data: {len(df_final)} rows, Range: {df_final['date'].min()} ~ {df_final['date'].max()}")

    # 5. Save
    save_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    df_final.to_parquet(save_path)
    print(f"Saved merged dataset to: {save_path}")

    # Also restore dataset_v4 for compatibility? No, let's stick to ml_dataset as primary now.
    # But for safety, keep the bak file.

if __name__ == "__main__":
    merge_datasets()
