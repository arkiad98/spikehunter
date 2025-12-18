import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

def verify_2024_performance():
    print(">> Verifying Model Performance on 2024 Data...")
    
    # 1. Load Model (Valid 2023 Model)
    model_path = "data/models/lgbm_model_2023.joblib"
    model = joblib.load(model_path)
    
    # 2. Load Data
    data_path = "data/proc/ml_dataset/ml_classification_dataset.parquet"
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. Filter 2024 Data (Test Set)
    test_df = df[df['date'] >= '2024-01-01'].copy()
    print(f">> 2024 Test Data: {len(test_df)} rows")
    
    # 4. Prepare Features & Target
    feature_names = model.feature_names_in_
    X_test = test_df[feature_names].fillna(0)
    y_test = test_df['label_class']
    
    # 5. Predict
    probs = model.predict_proba(X_test)[:, 1]
    
    # 6. Calculate Metrics
    ap = average_precision_score(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    
    print("\n" + "="*40)
    print(f" [2024 Out-of-Sample Performance]")
    print(f" Average Precision (AP): {ap:.4f}")
    print(f" ROC AUC: {auc:.4f}")
    print(f" Baseline (Random) AP: {y_test.mean():.4f}")
    print("="*40)

if __name__ == "__main__":
    verify_2024_performance()
