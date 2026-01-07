
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.collect_openapi import collect_openapi_index
from modules.utils_io import read_yaml

def test_openapi_progress():
    print("Testing OpenAPI Collection with Progress Bar...")
    
    # Read API Key
    settings_path = os.path.join(os.path.dirname(__file__), '../../config/settings.yaml')
    cfg = read_yaml(settings_path)
    api_key = cfg.get('ml_params', {}).get('krx_api_key')
    
    if not api_key:
        print("ERROR: API Key not found in settings.")
        return

    start_date = "2026-01-01"
    end_date = "2026-01-05" # 5 days
    
    print(f"Target Period: {start_date} ~ {end_date}")
    
    try:
        df = collect_openapi_index(start_date, end_date, api_key)
        
        print("\nCollection Result:")
        print(df)
        print(f"Rows Collected: {len(df)}")
        
        if len(df) > 0:
            print("SUCCESS: Data collected.")
        else:
            print("WARNING: No data returned (Check API or Market Holidays).")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_openapi_progress()
