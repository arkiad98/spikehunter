from pykrx import stock
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.patch_pykrx import patch_pykrx_referer

# Apply Patch
patch_pykrx_referer()

def test_partial():
    today = datetime.now().strftime("%Y%m%d")
    # For testing, use a recent valid trading day if today is weekend/early
    # 2026-01-06 is likely valid given the simulation context? 
    # Wait, the user's time is 2026!
    # Let's use 20260105 or 20260106.
    target_date = "20260105" 

    print(f"Testing Data Collection for {target_date}...\n")

    # 1. Test Stock Data (KOSPI)
    print("[1] Testing Stock Data (get_market_ohlcv)...")
    try:
        # 005930 = Samsung Electronics
        df_stock = stock.get_market_ohlcv(target_date, target_date, "005930")
        if not df_stock.empty:
            print(">>> SUCCESS: Stock Data Collected.")
            print(df_stock.head())
        else:
            print(">>> FAILED: Empty DataFrame.")
    except Exception as e:
        print(f">>> FAILED: {e}")

    print("\n" + "-"*30 + "\n")

    # 2. Test Index Data (KOSPI)
    print("[2] Testing Index Data (get_index_ohlcv)...")
    try:
        # 1001 = KOSPI
        df_index = stock.get_index_ohlcv(target_date, target_date, "1001")
        if not df_index.empty:
            print(">>> SUCCESS: Index Data Collected.")
            print(df_index.head())
        else:
            print(">>> FAILED: Empty DataFrame.")
    except Exception as e:
        print(f">>> FAILED: {e}")

if __name__ == "__main__":
    test_partial()
