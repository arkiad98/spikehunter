import pandas as pd
import numpy as np
from modules.features import generate_features

def test_feature_generation():
    print("Testing feature generation...")
    # Create dummy data
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = {
        'date': dates,
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 105,
        'low': np.random.rand(100) * 95,
        'close': np.random.rand(100) * 100,
        'volume': np.random.randint(100, 1000, 100)
    }
    df = pd.DataFrame(data)
    
    # Generate features
    try:
        df_feat = generate_features(df)
        print("Feature generation successful.")
        print(f"Generated DataFrame columns: {len(df_feat.columns)}")
        
        # Check for removed columns
        removed_cols = ['golden_cross_score', 'ma_alignment']
        for col in removed_cols:
            if col in df_feat.columns:
                print(f"[FAIL] Column '{col}' still exists!")
            else:
                print(f"[PASS] Column '{col}' successfully removed.")
                
    except Exception as e:
        print(f"[ERROR] Feature generation failed: {e}")

if __name__ == "__main__":
    test_feature_generation()
