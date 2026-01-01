import pandas as pd
import os

# Paths provided by user
PATH_OLD = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20251231_204537\wf_tradelog.csv"
PATH_NEW = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260101_001942\wf_tradelog.csv"

def analyze_wf(path, label):
    if not os.path.exists(path):
        print(f"[{label}] File not found: {path}")
        return None
        
    df = pd.read_csv(path)
    # df columns usually: entry_date, exit_date, profit_rate, capital...
    
    # Calculate global metrics
    win_rate = (df['return'] > 0).mean() * 100
    avg_profit = df['return'].mean() * 100
    total_trades = len(df)
    
    # MDD Calculation (approximate from trade sequence if equity curve not available, 
    # but here we just look at trade stats for simplicity first)
    # Check for consecutive losses
    
    print(f"--- {label} ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Profit: {avg_profit:.2f}%")
    print(f"Max Profit: {df['return'].max()*100:.2f}%")
    print(f"Max Loss: {df['return'].min()*100:.2f}%")
    
    # Period Analysis if 'period' column exists, else interpret from dates
    # Assuming 'entry_date' exists
    if 'entry_date' in df.columns:
        df['year'] = pd.to_datetime(df['entry_date']).dt.year
        yearly = df.groupby('year')['return'].mean() * 100
        print("\n[Yearly Avg Profit]")
        print(yearly)

    return df

print(">>> Comparing WF Backtest Results <<<\n")
df_old = analyze_wf(PATH_OLD, "OLD (Commit Ver)")
print("\n" + "="*40 + "\n")
df_new = analyze_wf(PATH_NEW, "NEW (Local Ver)")
