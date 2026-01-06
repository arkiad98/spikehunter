import sys
from pykrx import stock
from datetime import datetime, timedelta

def test_pykrx():
    print(f"Testing pykrx connection at {datetime.now()}")
    
    # Try to fetch KOSPI OHLCV for the last valid business day
    # Assuming 'today' might need data, but let's just pick a known recent date or today.
    # If today is weekend, stock.get_market_ohlcv returns empty or error depending on version maybe?
    # Let's try a date range for index, which usually returns something if working.
    
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
    
    print(f"Fetching KOSPI index data from {start_date} to {end_date}...")
    
    try:
        # Ticker 1001 is KOSPI
        df = stock.get_index_ohlcv(start_date, end_date, "1001")
        
        if df is None or df.empty:
            print("FAILED: Result is empty (potentially blocked or parsing error).")
            return False
        else:
            print(f"SUCCESS: Fetched {len(df)} rows.")
            print(df.tail(1))
            return True
            
    except Exception as e:
        print(f"FAILED: Exception occurred: {e}")
        return False

if __name__ == "__main__":
    success = test_pykrx()
    sys.exit(0 if success else 1)
