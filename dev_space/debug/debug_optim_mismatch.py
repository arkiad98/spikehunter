
import os
import joblib
import pandas as pd
from modules.utils_io import read_yaml

def check_feature_mismatch():
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]
    
    # Load Model
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    model = joblib.load(model_path)
    feature_names = list(getattr(model, 'feature_names_in_', []))
    print(f"Model expects {len(feature_names)} features.")
    print(f"Sample features: {feature_names[:10]}")
    
    # Load Data
    dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
    
    df = pd.read_parquet(dataset_path)
    print(f"Dataset columns (Sample): {list(df.columns)[:10]}")
    
    # Simulate Optimization Logic
    rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df_renamed = df.rename(columns=rename_map)
    
    missing_original = [c for c in feature_names if c not in df.columns]
    missing_renamed = [c for c in feature_names if c not in df_renamed.columns]
    
    print(f"Missing in Original DF: {len(missing_original)} -> {missing_original}")
    print(f"Missing in Renamed DF : {len(missing_renamed)} -> {missing_renamed}")
    
    if len(missing_renamed) > len(missing_original):
        print("!! CRITICAL: Renaming caused feature mismatch !!")

if __name__ == "__main__":
    check_feature_mismatch()
