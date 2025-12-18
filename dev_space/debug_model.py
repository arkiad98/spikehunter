import joblib
import pandas as pd
import numpy as np
import os
import sys

def debug_model():
    model_path = "data/models/lgbm_model.joblib"
    print(f"Checking model at: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file does not exist!")
        return

    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully. Type: {type(model)}")
        
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature names found: {len(model.feature_names_in_)}")
            print(f"First 5 features: {model.feature_names_in_[:5]}")
            
            # Create dummy data
            dummy_data = pd.DataFrame(np.random.rand(5, len(model.feature_names_in_)), columns=model.feature_names_in_)
            print("Attempting prediction on dummy data...")
            probs = model.predict_proba(dummy_data)[:, 1]
            print(f"Prediction successful. First 5 probs: {probs}")
            
        else:
            print("WARNING: 'feature_names_in_' attribute missing!")
            if hasattr(model, 'feature_name_'):
                 print(f"Found 'feature_name_' instead: {len(model.feature_name_)}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()
