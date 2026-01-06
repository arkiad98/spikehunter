
import pandas as pd
import numpy as np
import os
from modules.utils_io import read_yaml

def analyze_market_cycles():
    print(">> Analyzing Market Regime Cycles (KOSPI)...")
    
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    
    # 1. Load Market Index Data
    index_path = os.path.join(cfg['paths']['raw_index'], 'kospi.parquet')
    if not os.path.exists(index_path):
        index_path = os.path.join(cfg['paths']['raw_index'], 'kospi_index.csv')
    
    if not os.path.exists(index_path):
        print("KOSPI data not found.")
        return

    try:
        if index_path.endswith('.parquet'):
            df = pd.read_parquet(index_path)
        else:
            df = pd.read_csv(index_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Standardize column names
    if 'date' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns: df.rename(columns={'index': 'date'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    col = None
    for c in ['kospi_close', 'close', 'Close']:
        if c in df.columns:
            col = c
            break
            
    if col is None:
        print("Close column not found.")
        return
        
    close = df[col]
    
    # 2. Define Regimes using technical indicators
    # Method: MA200 & DD (Bull/Bear definition)
    # Bull: Price > MA200
    # Bear: Price < MA200 and DD < -20% from peak? 
    # Let's use a simpler, standard classification:
    # Bull: SMA50 > SMA200 (Golden Cross state)
    # Bear: SMA50 < SMA200 (Dead Cross state)
    
    df['ma50'] = close.rolling(50).mean()
    df['ma200'] = close.rolling(200).mean()
    
    # Drop NaN
    df = df.dropna().copy()
    
    df['regime'] = np.where(df['ma50'] > df['ma200'], 'Bull', 'Bear')
    
    # Further refinement: Sideways?
    # If slope of MA200 is flat?
    # For now, let's stick to Bull/Bear cycles to find major inflection points.
    
    # 3. Calculate Cycle Durations
    df['cycle_id'] = (df['regime'] != df['regime'].shift(1)).cumsum()
    
    cycles = df.groupby(['cycle_id', 'regime']).agg(
        start_date=('date', 'min'),
        end_date=('date', 'max'),
        duration_days=('date', lambda x: (x.max() - x.min()).days)
    ).reset_index()
    
    print("\n[Market Cycles (SMA50 vs SMA200)]")
    print(f"{'Regime':<10} | {'Start':<12} | {'End':<12} | {'Days':<5}")
    print("-" * 50)
    for _, row in cycles.iterrows():
        print(f"{row['regime']:<10} | {row['start_date'].strftime('%Y-%m-%d')}   | {row['end_date'].strftime('%Y-%m-%d')}   | {row['duration_days']:<5}")
        
    print("\n[Summary Statistics]")
    print(cycles.groupby('regime')['duration_days'].describe())
    
    avg_total_cycle = cycles['duration_days'].mean()
    print(f"\nAverage Regime Duration: {avg_total_cycle:.1f} days ({avg_total_cycle/30:.1f} months)")
    
    # Recommendation
    print("\n[Analysis]")
    print(f"Cycle Count: {len(cycles)}")
    
    # If using 2-year lookback (730 days), does it cover at least one full cycle (Bull+Bear)?
    # Typically a full market cycle is Bull + Bear.
    
    bull_avg = cycles[cycles['regime']=='Bull']['duration_days'].mean()
    bear_avg = cycles[cycles['regime']=='Bear']['duration_days'].mean()
    
    print(f"Avg Bull Duration: {bull_avg:.1f} days")
    print(f"Avg Bear Duration: {bear_avg:.1f} days")
    print(f"Full Cycle (Bull+Bear) Approx: {bull_avg + bear_avg:.1f} days ({(bull_avg + bear_avg)/365:.1f} years)")
    
    if (bull_avg + bear_avg) < 365 * 2:
        print(">> 2-Year (730 days) lookback covers significantly more than one full cycle.")
        print(">> This implies high safety but slower adaptation.")
    else:
        print(">> 2-Year lookback might be shorter than a full cycle.")
        print(">> Risk: Training only on Bull market then testing on Bear market.")

if __name__ == "__main__":
    analyze_market_cycles()
