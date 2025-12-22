
import pandas as pd
import os
from modules.utils_io import read_yaml

def analyze_labels():
    cfg = read_yaml("config/settings.yaml")
    dataset_path = os.path.join(cfg["paths"]["ml_dataset"], "ml_classification_dataset.parquet")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    df = pd.read_parquet(dataset_path)
    total = len(df)
    positives = df['label_class'].sum()
    negatives = total - positives
    ratio = positives / total

    print(f"Total Samples: {total}")
    print(f"Positive Labels (1): {positives} ({ratio*100:.2f}%)")
    print(f"Negative Labels (0): {negatives} ({(1-ratio)*100:.2f}%)")
    
    # Check if scale_pos_weight matches the ratio
    # Ideally scale_pos_weight ~= negatives / positives
    ideal_weight = negatives / positives if positives > 0 else 0
    current_weight = cfg["ml_params"]["lgbm_params_classification"].get("scale_pos_weight", 1.0)
    
    print(f"\nIdeal scale_pos_weight (Balanced): {ideal_weight:.4f}")
    print(f"Current scale_pos_weight (Settings): {current_weight}")
    
    if abs(current_weight - ideal_weight) > 1.0:
        print("WARNING: Significant mismatch in class weights.")

if __name__ == "__main__":
    analyze_labels()
