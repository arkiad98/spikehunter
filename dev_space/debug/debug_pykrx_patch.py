import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.utils_logger import setup_global_logger

# Setup Logger
setup_global_logger("debug_pykrx")
logger = logging.getLogger(__name__)

def test_patch_and_fetch():
    print(">>> 1. Importing modules...")
    from pykrx import stock
    from modules.patch_pykrx import patch_pykrx_referer
    
    print(">>> 2. Applying Patch...")
    patch_pykrx_referer()
    
    target_date = "20260102"
    print(f">>> 3. Fetching KOSPI data for {target_date}...")
    
    try:
        df = stock.get_market_ohlcv_by_ticker(target_date, "KOSPI")
        
        print(f"\n[Result] DataFrame Shape: {df.shape}")
        if not df.empty:
            print(f"[Result] Columns: {df.columns.tolist()}")
            print("[Result] Head(3):")
            print(df.head(3))
        else:
            print("[Result] DataFrame is EMPTY.")
            
        # Check if patch specific logic worked (Renamed columns)
        expected_cols = ['시가', '고가', '저가', '종가']
        if all(c in df.columns for c in expected_cols):
             print("\n[SUCCESS] Column renaming successful. Patch IS working.")
        else:
             print("\n[FAILURE] Required columns missing. Patch might NOT be working.")
             
    except Exception as e:
        print(f"\n[ERROR] Fetch failed: {e}")

if __name__ == "__main__":
    test_patch_and_fetch()
