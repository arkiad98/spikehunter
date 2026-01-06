import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Apply Patch
from modules.patch_pykrx import patch_pykrx_referer
patch_pykrx_referer()

from modules.collect import run_collect
from pykrx import stock
print(f"Verify: stock.get_market_ohlcv_by_ticker is {stock.get_market_ohlcv_by_ticker}")
if "patched" in str(stock.get_market_ohlcv_by_ticker):
    print("Verify: PATCHED CONFIRMED")
else:
    print("Verify: PATCH NOT APPLIED?")

def verify_collection():
    print("Verifying data collection for 2026-01-06...")
    try:
        # Run for a single day to test connectivity
        # Note: settings.yaml path is relative to project root
        settings_path = os.path.join(os.path.dirname(__file__), '../../config/settings.yaml')
        settings_path = os.path.abspath(settings_path)
        
        print(f"Settings path: {settings_path}")
        
        # Force overwrite to ensure it tries to fetch
        result = run_collect(settings_path, start="20260106", end="20260106", overwrite=True)
        
        if result:
            print("Collection SUCCESS returned True")
        else:
            print("Collection FAILED returned False")
    except Exception as e:
        print(f"Error during collection: {e}")

if __name__ == "__main__":
    verify_collection()
