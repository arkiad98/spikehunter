import sys
import os

# Add project root to path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from modules.patch_pykrx import patch_pykrx_referer
patch_pykrx_referer()

from pykrx import stock
import datetime

def test_pykrx_fetch():
    print("Testing Pykrx data fetch...")
    today = datetime.datetime.now().strftime("%Y%m%d")
    try:
        # Try fetching KOSPI index
        df = stock.get_index_ohlcv(today, today, "1001")
        print("Success: Fetched Index")
        print(df.head())
        
        # Try fetching stock price (Samsung Electronics)
        df_stock = stock.get_market_ohlcv(today, today, "005930")
        print("Success: Fetched Stock")
        print(df_stock.head())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed: {repr(e)}")

if __name__ == "__main__":
    test_pykrx_fetch()
