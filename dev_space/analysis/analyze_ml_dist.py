import pandas as pd
import joblib
import os
import numpy as np
from modules.utils_io import read_yaml

def analyze_scores():
    cfg = read_yaml("config/settings.yaml")
    paths = cfg["paths"]
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset_path = os.path.join(paths["features"], "dataset_v4.parquet")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(paths["ml_dataset"], "ml_classification_dataset.parquet")
        
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Dataset Loaded: {len(df)} rows, Period: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # 2. Load Model
    print("Loading Model...")
    model_path = os.path.join(paths["models"], "lgbm_model.joblib")
    model = joblib.load(model_path)
    
    # 3. Predict Scores
    print("Calculating ML Scores...")
    feature_names = model.feature_names_in_
    X = df[feature_names].fillna(0)
    scores = model.predict_proba(X)[:, 1]
    df['ml_score'] = scores
    
    # 4. Analyze by Year
    print("\n" + "="*50)
    print(" [ML Score Statistics by Year]")
    print(f" {'Year':<6} | {'Count':<8} | {'Mean':<8} | {'Max':<8} | {'>0.52':<6}")
    print("-" * 50)
    
    for year in sorted(df['date'].dt.year.unique()):
        sub = df[df['date'].dt.year == year]
        score = sub['ml_score']
        over_thresh = (score > 0.52).sum()
        print(f" {year:<6} | {len(sub):<8} | {score.mean():.4f}   | {score.max():.4f}   | {over_thresh:<6}")
        
    print("="*50)
    
    # 5. Analyze Recent Period (2025-06-30 ~ 2025-12-30)
    start_dt = pd.Timestamp("2025-06-30")
    end_dt = pd.Timestamp("2025-12-30")
    recent = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    
    print("\n [Recent Period Analysis: 2025-06-30 ~ 2025-12-30]")
    if not recent.empty:
        r_score = recent['ml_score']
        print(f" Count: {len(recent)}")
        print(f" Mean: {r_score.mean():.4f}")
        print(f" Max:  {r_score.max():.4f}")
        print(f" > 0.52: {(r_score > 0.52).sum()}")
    else:
        print(" No data in recent period.")
        
    print("\nConclusion:")
    max_all = df['ml_score'].max()
    max_recent = recent['ml_score'].max() if not recent.empty else 0
    print(f"Entire Period Max: {max_all:.4f}")
    print(f"Recent Period Max: {max_recent:.4f}")
    
    if max_all > 0.52 and max_recent < 0.52:
        print(">> Optimization found high scores in earlier years, picking threshold 0.52.")
        print(">> But recent market conditions yield lower scores, causing 0 trades.")

if __name__ == "__main__":
    analyze_scores()
