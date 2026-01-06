import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.collect_openapi import collect_openapi_index
from datetime import datetime, timedelta
import pandas as pd

# Key from Settings
KEY = "9366E92B67004AC7889ED857B0A2C589C8A69D1B"

def verify():
    # Test only 1 day to be fast
    start = datetime(2025, 1, 3) 
    end = datetime(2025, 1, 3)
    
    print(f"Testing OpenAPI (Debug URL) with Key: {KEY[:5]}...")
    df = collect_openapi_index(start, end, KEY)
    
    print("\nResult:")
    if not df.empty:
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
    else:
        print("Empty DataFrame returned.")
        print("(Note: This might be due to Authorization Pending state or wrong column mapping)")

if __name__ == "__main__":
    verify()
