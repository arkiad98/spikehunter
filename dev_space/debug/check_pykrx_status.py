
import pykrx
from pykrx import stock
import datetime
import pandas as pd

try:
    print(f"Pykrx Version: {pykrx.__version__}")
except:
    print("Could not find __version__")

try:
    # Try fetching KOSPI index for a 2026 date
    target_date = "20260105"
    print(f"Attempting to fetch index for {target_date}...")
    df = stock.get_index_ohlcv(target_date, target_date, "1001")
    print("Result (Index):")
    print(df)
    
    # Try fetching stock ticker for 2026
    print(f"Attempting to fetch stock tickers for {target_date}...")
    tickers = stock.get_market_ticker_list(target_date, market="KOSPI")
    print(f"Result (Tickers): {len(tickers)} found")
    if len(tickers) > 0:
        print(f"Sample: {tickers[:5]}")
        
except Exception as e:
    print(f"Error: {e}")
