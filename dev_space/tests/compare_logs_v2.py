
import pandas as pd
import numpy as np

wf_path = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260102_130851\wf_tradelog.csv"
std_path = r"D:\spikehunter\data\proc\backtest\run_20260102_133019\tradelog.parquet"

def calculate_metrics(df, name):
    if df.empty:
        return {'Name': name, 'Trades': 0}
    
    df = df.copy()
    if 'entry_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df = df.sort_values('entry_date')
    
    total_trades = len(df)
    win_rate = (df['return'] > 0).mean() * 100
    avg_return = df['return'].mean()
    
    # Cumulative return (simple compounding estimation)
    # Assuming valid trades are sequential or parallel in a way that allows simple compounding for approx.
    # For accurate MDD and CAGR, we need daily equity, but specific trade sum is okay for comparison
    # We'll use sum of returns for simple ROI estimation or prod(1+r)
    total_return = (1 + df['return']).prod() - 1
    
    # Calculate Sharpe Ratio (proxy)
    # Assuming daily returns logic, but we have trade returns. 
    # Just use Avg / Std of trade returns as a "Trade Sharpe"
    trade_std = df['return'].std()
    trade_sharpe = avg_return / trade_std if trade_std != 0 else 0
    
    # MDD is hard without equity curve, but we can approximate it from cumulative return series of trades
    equity = (1 + df['return']).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = drawdown.min() * 100
    
    start_date = df['entry_date'].min()
    end_date = df['entry_date'].max()
    
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else 0
    cagr = ((total_return + 1) ** (1/years) - 1) * 100 if years > 0 else 0

    return {
        'Name': name,
        'Start': start_date.strftime('%Y-%m-%d'),
        'End': end_date.strftime('%Y-%m-%d'),
        'Trades': total_trades,
        'Win Rate %': win_rate,
        'Avg Return %': avg_return * 100,
        'Total Return %': total_return * 100,
        'CAGR %': cagr,
        'MDD %': mdd,
        'Trade Sharpe': trade_sharpe
    }

try:
    print("Loading data...")
    df_wf = pd.read_csv(wf_path)
    df_wf['entry_date'] = pd.to_datetime(df_wf['entry_date'])
    
    df_std = pd.read_parquet(std_path)
    df_std['entry_date'] = pd.to_datetime(df_std['entry_date'])
    
    # Full Stats
    stats_wf = calculate_metrics(df_wf, "WF (Full)")
    stats_std = calculate_metrics(df_std, "Standard (Full)")
    
    # Filter Standard to WF period
    start_date = df_wf['entry_date'].min()
    end_date = df_wf['entry_date'].max()
    
    df_std_filtered = df_std[(df_std['entry_date'] >= start_date) & (df_std['entry_date'] <= end_date)]
    stats_std_filtered = calculate_metrics(df_std_filtered, "Standard (Matched Period)")
    
    results = pd.DataFrame([stats_wf, stats_std, stats_std_filtered])
    
    # Formatting
    format_cols = ['Win Rate %', 'Avg Return %', 'Total Return %', 'CAGR %', 'MDD %', 'Trade Sharpe']
    for col in format_cols:
        results[col] = results[col].apply(lambda x: f"{x:.2f}")
        
    print("\n--- Comparative Analysis ---")
    print(results.to_string(index=False))

except Exception as e:
    print(f"Error: {e}")
