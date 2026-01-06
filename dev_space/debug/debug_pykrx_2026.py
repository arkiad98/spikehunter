from pykrx import stock
import pandas as pd
import traceback
from unittest.mock import patch

print(">>> Debugging stock.get_market_ohlcv for 20260102 (KOSPI)")
try:
    df = stock.get_market_ohlcv("20260102", market="KOSPI")
    print("Result Dataframe Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Head:", df.head(2))
except Exception:
    traceback.print_exc()

print("\n>>> Debugging stock.get_index_ohlcv for 1001 (KOSPI)")
try:
    # Attempt reproduction
    df_idx = stock.get_index_ohlcv("20260101", "20260102", "1001")
    print("Result Index DF:", df_idx)
except Exception:
    print("Reproduction successful (Error caught):")
    # traceback.print_exc() # Too verbose, we know it fails

print("\n>>> Testing Monkeypatch for get_index_ohlcv")

# Monkeypatch attempt
def mock_get_index_ticker_name(ticker):
    print(f"Mock called for ticker: {ticker}")
    if str(ticker) == "1001":
        return "코스피"
    return "Unknown"

# We need to find where to patch. stock.stock_api.get_index_ticker_name
try:
    with patch('pykrx.stock.stock_api.get_index_ticker_name', side_effect=mock_get_index_ticker_name):
        print("Monkeypatch active. Retrying...")
        df_idx = stock.get_index_ohlcv("20260101", "20260102", "1001")
        print("Success! Result Index DF:")
        print(df_idx)
except Exception:
    traceback.print_exc()
