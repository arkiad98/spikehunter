from pykrx import stock
from datetime import datetime, timedelta
import traceback

print("Current Date:", datetime.now())

try:
    now = datetime.now()
    end_str = now.strftime("%Y%m%d")
    start_str = (now - timedelta(days=14)).strftime("%Y%m%d")
    print(f"Fetching Samsung Electronics (005930) from {start_str} to {end_str}")
    
    # Use Samsung Electronics (005930) as proxy for business days
    df = stock.get_market_ohlcv(start_str, end_str, "005930")
    print("Result:")
    print(df)
    
    if not df.empty:
        print("Success! Business days found.")
        print(df.index)
    else:
        print("Failed: Empty dataframe")

except Exception:
    traceback.print_exc()
