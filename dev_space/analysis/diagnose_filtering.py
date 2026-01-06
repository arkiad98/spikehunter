
import pandas as pd
import os

import joblib
import numpy as np

def diagnose():
    # settings from user log
    min_ml_score = 0.4497
    min_mfi = 52.67
    max_market_vol = 0.0349
    
    path = "d:/spikehunter/data/proc/ml_dataset/ml_classification_dataset.parquet"
    model_path = "d:/spikehunter/data/models/lgbm_model.joblib"
    
    if not os.path.exists(path):
        print("Dataset not found")
        return
    if not os.path.exists(model_path):
        print("Model not found")
        return

    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load Model
    print(f"Loading Model {model_path}...")
    model = joblib.load(model_path)
    
    # Predict
    features = model.feature_names_in_
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"CRITICAL: {len(missing)} features missing in dataset!")
        print(f"Missing (example): {missing[:10]}")
    
    X = df[features].fillna(0)
    
    print("Generating ML Scores...")
    # debug print first row of X
    print("X head:", X.iloc[0].to_dict())
    
    # Check predictions
    if hasattr(model, "predict_proba"):
        raw_preds = model.predict_proba(X)[:, 1]
    else:
        raw_preds = model.predict(X)
        
    print(f"Raw Predictions Stats: Min={raw_preds.min():.4f}, Max={raw_preds.max():.4f}, Mean={raw_preds.mean():.4f}")
    print(f"Unique Values: {np.unique(raw_preds)[:10]} ...")
    
    df['ml_score'] = raw_preds
    
    # Filter for Test Period
    start_date = "2025-06-30"
    end_date = "2025-12-30"
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    print(f"\nAnalyzing Period: {start_date} ~ {end_date}")
    print(f"Total Rows: {len(df)}")
    
    # 1. ML Score Filter
    candidates = df[df['ml_score'] >= min_ml_score]
    print(f"1. Candidates with Score >= {min_ml_score}: {len(candidates)} ({len(candidates)/len(df)*100:.2f}%)")
    
    if candidates.empty:
        print("   -> STOP: ML Score threshold is too high for this period.")
        print(f"   -> Max ML Score in period: {df['ml_score'].max()}")
        return

    # 2. Volatility Filter
    passed_vol = candidates[candidates['market_volatility'] <= max_market_vol]
    print(f"2. Passed Volatility <= {max_market_vol}: {len(passed_vol)} (Rejected: {len(candidates) - len(passed_vol)})")
    
    # 3. MFI Filter
    passed_mfi = passed_vol[passed_vol['mfi_14'] >= min_mfi]
    print(f"3. Passed MFI >= {min_mfi}: {len(passed_mfi)} (Rejected: {len(passed_vol) - len(passed_mfi)})")
    
    print("\n[Passed Candidates]")
    if not passed_mfi.empty:
        print(passed_mfi[['date', 'code', 'ml_score', 'mfi_14', 'market_volatility']].head(10))
    else:
        print("None.")

if __name__ == "__main__":
    diagnose()
