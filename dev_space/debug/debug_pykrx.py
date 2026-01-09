
from pykrx import stock
import pandas as pd

def debug_columns():
    date = "20240102" # Known valid date
    print(f"Fetching data for {date}...")
    try:
        df = stock.get_market_ohlcv_by_ticker(date, market="KOSPI")
        print("--- Columns ---")
        print(df.columns)
        print("--- Head ---")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_columns()
