import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

def verify_ap_gap():
    print(">> Verifying AP Gap (Train vs Test)...")
    
    # 1. Load Model (Valid 2023 Model)
    model_path = "data/models/lgbm_model_2023.joblib"
    if not joblib.os.path.exists(model_path):
        print("Model not found.")
        return
    model = joblib.load(model_path)
    
    # 2. Load Data
    data_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. Split Data
    # Train: 2023 (Last year of training) - to see how well it learned recent patterns
    train_df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-12-31')].copy()
    # Test: 2024 (Out of Sample)
    test_df = df[df['date'] >= '2024-01-01'].copy()
    
    print(f">> Train Set (2023): {len(train_df)} rows")
    print(f">> Test Set (2024): {len(test_df)} rows")
    
    feature_names = model.feature_names_in_
    
    # 4. Calculate Metrics
    results = {}
    for name, data in [("Train (2023)", train_df), ("Test (2024)", test_df)]:
        X = data[feature_names].fillna(0)
        y = data['label_class']
        
        if len(y) == 0: continue
        
        probs = model.predict_proba(X)[:, 1]
        ap = average_precision_score(y, probs)
        auc = roc_auc_score(y, probs)
        baseline = y.mean()
        
        results[name] = {"AP": ap, "AUC": auc, "Baseline": baseline}
        
    # 5. Report
    print("\n" + "="*50)
    print(f" {'Metric':<15} | {'Train (2023)':<15} | {'Test (2024)':<15} | {'Gap':<10}")
    print("-" * 50)
    
    train_ap = results["Train (2023)"]["AP"]
    test_ap = results["Test (2024)"]["AP"]
    gap = test_ap - train_ap
    
    print(f" {'AP':<15} | {train_ap:.4f}          | {test_ap:.4f}          | {gap:.4f}")
    print(f" {'Baseline':<15} | {results['Train (2023)']['Baseline']:.4f}          | {results['Test (2024)']['Baseline']:.4f}          |")
    print(f" {'Lift (vs Base)':<15} | {train_ap/results['Train (2023)']['Baseline']:.2f}x          | {test_ap/results['Test (2024)']['Baseline']:.2f}x          |")
    print("="*50)

if __name__ == "__main__":
    verify_ap_gap()
