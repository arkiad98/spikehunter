from pykrx import stock
from datetime import datetime, timedelta
import traceback

print("Current Date:", datetime.now())

try:
    now = datetime.now()
    end_str = now.strftime("%Y%m%d")
    start_str = (now - timedelta(days=14)).strftime("%Y%m%d")
    print(f"Fetching KOSPI from {start_str} to {end_str}")
    
    df = stock.get_index_ohlcv(start_str, end_str, "1001")
    print("Result:")
    print(df)
except Exception:
    traceback.print_exc()
