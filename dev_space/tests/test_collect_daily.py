
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.patch_pykrx import patch_pykrx_referer
patch_pykrx_referer()

from modules.collect import _fetch_and_merge_daily_data
from modules.collect_openapi import collect_openapi_index

REQUIRED_RAW_COLS = [
    "date", "code", "open", "high", "low", "close", "volume", "value",
    "inst_net_val", "foreign_net_val"
]

def test_daily_fetch():
    target_date = pd.Timestamp("2026-01-06")
    print(f"Testing daily fetch for {target_date.date()}...")
    try:
        df = _fetch_and_merge_daily_data(target_date, REQUIRED_RAW_COLS)
        print("Fetch result:")
        print(df.head())
        print("Columns:", df.columns)
        if 'inst_net_val' in df.columns and 'foreign_net_val' in df.columns:
            print("SUCCESS: Fund flow columns present.")
        else:
            print("FAILURE: Fund flow columns missing.")
            
        if df.empty:
            print("WARNING: DataFrame is empty (Market might be closed or API error)")
        else:
            print(f"Collected {len(df)} rows.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_daily_fetch()
