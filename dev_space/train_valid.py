import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from modules.utils_io import read_yaml, optimize_memory_usage
from modules.train import _get_core_features_from_registry
from modules.derive import _get_feature_cols

def train_valid_model():
    print(">> Training Valid Model (Data <= 2023-12-31)...")
    
    # 1. Load Config & Data
    cfg = read_yaml("config/settings.yaml")
    dataset_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. Split Data (Train <= 2023)
    train_df = df[df['date'] <= '2023-12-31'].copy()
    print(f">> Train Data: {len(train_df)} rows (Max Date: {train_df['date'].max()})")
    
    # 3. Features
    registry_path = "config/feature_registry.yaml"
    core_features = _get_core_features_from_registry(registry_path)
    available_cols = set(train_df.columns)
    feature_cols = [f for f in core_features if f in available_cols]
    if not feature_cols: feature_cols = _get_feature_cols(train_df.columns)
    
    print(f">> Features: {len(feature_cols)}")
    
    X_train = train_df[feature_cols]
    y_train = train_df['label_class']
    
    # 4. Train LightGBM
    lgbm_params = cfg['ml_params'].get('lgbm_params_classification', {})
    # Remove param_space keys if any
    lgbm_params = {k: v for k, v in lgbm_params.items() if not k.startswith('param_space_')}
    
    print(">> Training LightGBM...")
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train)
    
    # 5. Save Model
    save_path = "data/models/lgbm_model_2023.joblib"
    joblib.dump(model, save_path)
    print(f">> Model saved to {save_path}")

if __name__ == "__main__":
    train_valid_model()
