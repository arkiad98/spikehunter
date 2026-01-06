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

def test_index_scope():
    today = "20260105" 
    print(f"Testing Index Data Scope for {today}...\n")

    # 1. Test KOSPI Index OHLCV (Known to fail)
    print("[1] Testing KOSPI Index (1001) OHLCV...")
    try:
        df = stock.get_index_ohlcv(today, today, "1001")
        print(f">>> KOSPI Result: {'Success' if not df.empty else 'Empty'}")
    except Exception as e:
        print(f">>> KOSPI Failed: {e}")

    print("-" * 20)

    # 2. Test KOSDAQ Index OHLCV
    print("[2] Testing KOSDAQ Index (2001) OHLCV...")
    try:
        df = stock.get_index_ohlcv(today, today, "2001")
        print(f">>> KOSDAQ Result: {'Success' if not df.empty else 'Empty'}")
    except Exception as e:
        print(f">>> KOSDAQ Failed: {e}")

    print("-" * 20)

    # 3. Test Index Status/Fundamental (Alternative endpoint?)
    # get_index_fundamental uses a different scraping logic usually
    print("[3] Testing Index Fundamental (1001)...")
    try:
        df = stock.get_index_fundamental(today, today, "1001")
        print(f">>> Fundamental Result: {'Success' if not df.empty else 'Empty'}")
    except Exception as e:
        print(f">>> Fundamental Failed: {e}")
        
    print("-" * 20)

    # 4. Test Index Ticker List (Metadata)
    print("[4] Testing Index Ticker List (KOSPI)...")
    try:
        tickers = stock.get_index_ticker_list(today, market="KOSPI")
        print(f">>> Ticker List Result: {len(tickers)} tickers found.")
    except Exception as e:
        print(f">>> Ticker List Failed: {e}")


if __name__ == "__main__":
    test_index_scope()
