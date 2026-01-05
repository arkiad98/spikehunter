
import pandas as pd
import numpy as np
import os
from modules.utils_io import read_yaml

def main():
    print(">> Diagnosing Period 5 (2022-01-01 ~ 2023-12-31)...")
    
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    
    # 1. Load Market Index Data
    index_path = os.path.join(cfg['paths']['raw_index'], 'kospi.parquet')
    if not os.path.exists(index_path):
        # Fallback to csv
        index_path = os.path.join(cfg['paths']['raw_index'], 'kospi_index.csv')
        if not os.path.exists(index_path):
            print(f"Index file not found: {index_path}")
            return
        df_index = pd.read_csv(index_path)
    else:
        df_index = pd.read_parquet(index_path)
        
    df_index['date'] = pd.to_datetime(df_index['date'])
    df_index = df_index.sort_values('date')
    
    # Filter Period 5
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    mask = (df_index['date'] >= start_date) & (df_index['date'] <= end_date)
    print(f"Columns: {df_index.columns}")
    
    # Rename if needed
    if 'close' in df_index.columns and 'kospi_close' not in df_index.columns:
        df_index.rename(columns={'close': 'kospi_close'}, inplace=True)
        
    df_p5 = df_index[mask].copy()
    
    # 2. Calculate Volatility (Same logic as features.py)
    # market_volatility = close.pct_change().rolling(20).std()
    
    # df_p5['kospi_close'] is now guaranteed or we fail gracefully
    if 'kospi_close' not in df_p5.columns:
        print("Error: 'kospi_close' column not found.")
        return

    df_p5['market_volatility'] = df_p5['kospi_close'].pct_change().rolling(20).std()
    
    print("\n[Market Volatility (std20) Statistics for Period 5]")
    print(df_p5['market_volatility'].describe())
    
    # Check Strategy Thresholds
    # Current setting: max_market_vol low: 0.015, high: 0.04
    threshold = 0.04
    
    over_threshold = df_p5[df_p5['market_volatility'] > threshold]
    ratio = len(over_threshold) / len(df_p5) * 100
    
    print(f"\n[Filter Check]")
    print(f"Days with Volatility > {threshold}: {len(over_threshold)} / {len(df_p5)} ({ratio:.2f}%)")
    
    print(f"\nAvg Volatility: {df_p5['market_volatility'].mean():.4f}")
    print(f"Max Volatility: {df_p5['market_volatility'].max():.4f}")
    
    if df_p5['market_volatility'].mean() > 0.015:
        print("\n>> Analysis: Average volatility is consistent with bear markets.")
    
    if len(over_threshold) > 0:
        print(">> Warning: Some days exceed the max threshold (0.04).")
        
    # Also check min_mfi logic?
    # Strategy generally requires MFI > min_mfi (e.g., > 40)
    # If market MFI is low, it might filter out too many.
    
    print("\nDiagnosis Complete.")

if __name__ == "__main__":
    main()
