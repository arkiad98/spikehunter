import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

import logging

# 1. Import modules to trigger the monkeypatch
print("Importing modules to trigger monkeypatch...")
try:
    import modules.collect  # Should apply patch
    import modules.predict  # Should apply patch
    print("Modules imported successfully.")
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

from pykrx import stock

# Configure logging
logging.basicConfig(level=logging.INFO)

def verify_fix():
    print("\n--- 1. Direct pykrx Test (Index OHLCV) ---")
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
    print(f"Fetching KOSPI from {start_date} to {end_date}...")
    
    try:
        df = stock.get_index_ohlcv(start_date, end_date, "1001")
        if df is None or df.empty:
            print("FAILED: Result is empty.")
        else:
            print(f"SUCCESS: Fetched {len(df)} rows.")
            print(df.tail(1))
    except Exception as e:
        print(f"FAILED: Exception: {e}")

    print("\n--- 2. Application Logic Test (get_target_business_day) ---")
    try:
        target_date = modules.predict.get_target_business_day()
        if target_date:
            print(f"SUCCESS: Target Business Day: {target_date}")
        else:
            print("FAILED: Returned None")
    except Exception as e:
        print(f"FAILED: Error in logic: {e}")

if __name__ == "__main__":
    verify_fix()
