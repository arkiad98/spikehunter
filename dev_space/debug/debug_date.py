from pykrx import stock
from datetime import datetime, timedelta

now = datetime.now()
start_str = (now - timedelta(days=5)).strftime("%Y%m%d")
end_str = now.strftime("%Y%m%d")

print(f"Checking data from {start_str} to {end_str}")
try:
    df = stock.get_index_ohlcv(start_str, end_str, "1001")
    print("Result DataFrame:")
    print(df)
    print(f"Last Index: {df.index[-1]}")
except Exception as e:
    print(f"Error: {e}")
