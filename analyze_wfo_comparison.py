
import pandas as pd
import numpy as np
import os
import glob

def calculate_metrics(df_equity):
    """Calculate key metrics from daily equity series"""
    if df_equity.empty: return {}
    
    # Identify equity column
    col = None
    for c in ['total_equity', 'equity', 'value', 'total_value', 'TotalValue']:
        if c in df_equity.columns:
            col = c
            break
            
    if col is None:
        print(f"Error: Equity column not found. Available: {df_equity.columns}")
        return {}
        
    df_equity['total_equity'] = df_equity[col] # Standardize
    
    # CAGR
    days = (df_equity['date'].iloc[-1] - df_equity['date'].iloc[0]).days
    if days == 0: return {}
    final_equity = df_equity['total_equity'].iloc[-1]
    initial_equity = df_equity['total_equity'].iloc[0]
    cagr = (final_equity / initial_equity) ** (365 / days) - 1
    
    # MDD
    df_equity['peak'] = df_equity['total_equity'].cummax()
    df_equity['dd'] = (df_equity['total_equity'] - df_equity['peak']) / df_equity['peak']
    mdd = df_equity['dd'].min()
    
    # Sharpe
    df_equity['daily_ret'] = df_equity['total_equity'].pct_change().fillna(0)
    sharpe = df_equity['daily_ret'].mean() / (df_equity['daily_ret'].std() + 1e-9) * np.sqrt(252)
    
    return {
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe,
        'Final_Equity': final_equity,
        'Days': days
    }

def analyze_path(path, label):
    print(f"\n>> Analyzing: {label}")
    print(f"Path: {path}")
    
    # Load Equity
    equity_path = os.path.join(path, "wf_daily_equity.parquet")
    if not os.path.exists(equity_path):
        print("Equity file not found.")
        return None
        
    df_equity = pd.read_parquet(equity_path)
    if 'date' not in df_equity.columns:
        df_equity = df_equity.reset_index()
        if 'index' in df_equity.columns:
             df_equity.rename(columns={'index': 'date'}, inplace=True)
             
    # Ensure date is datetime
    if 'date' in df_equity.columns:
        df_equity['date'] = pd.to_datetime(df_equity['date'])
    else:
        print("Error: Date column not found even after reset_index")
        print(df_equity.columns)
        return {}
    
    stats = calculate_metrics(df_equity)
    print(f"  [Overall] CAGR: {stats['CAGR']:.2%} | MDD: {stats['MDD']:.2%} | Sharpe: {stats['Sharpe']:.2f}")
    
    # Period Analysis
    # We don't have explicit period dates in metrics, but we can look at chunks or years
    # 2024-01~2024-06 is Period 5 Test
    
    df_p5 = df_equity[(df_equity['date'] >= '2024-01-01') & (df_equity['date'] <= '2024-06-30')].copy()
    stats_p5 = calculate_metrics(df_p5)
    
    print(f"  [Period 5] 24.01~24.06 (Test): Return: {(stats_p5['Final_Equity']/df_p5['total_equity'].iloc[0]-1):.2%} | MDD: {stats_p5['MDD']:.2%}")
    
    # Load Tradelog for count
    tradelog_path = os.path.join(path, "wf_tradelog.csv")
    if os.path.exists(tradelog_path):
        df_trade = pd.read_csv(tradelog_path)
        print(f"  [Trade] Total Count: {len(df_trade)}")
        
        # Period 5 Trades
        df_trade['entry_date'] = pd.to_datetime(df_trade['entry_date'])
        
        # Identify profit column
        p_col = None
        for c in ['profit_rate', 'return', 'roi', 'pnl_pct', 'profit_pct']:
            if c in df_trade.columns:
                p_col = c
                break
        
        if p_col:
            p5_trades = df_trade[(df_trade['entry_date'] >= '2024-01-01') & (df_trade['entry_date'] <= '2024-06-30')]
            print(f"  [Period 5] Trade Count: {len(p5_trades)} | Avg Profit: {p5_trades[p_col].mean():.2%}")
        else:
             print(f"  [Error] Profit column not found. Available: {df_trade.columns}")
        
    return stats, stats_p5

def main():
    path_old = r"d:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260104_153447"
    path_new = r"d:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260104_173043"
    
    s1, p1 = analyze_path(path_old, "Old (Strict MDD)")
    s2, p2 = analyze_path(path_new, "New (Relaxed MDD)")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<15} | {'Old':<15} | {'New':<15} | {'Diff'}")
    print("-" * 60)
    
    # Overall
    diff_cagr = s2['CAGR'] - s1['CAGR']
    diff_mdd = s2['MDD'] - s1['MDD'] # Negative is bad if MDD gets deeper (e.g. -20 -> -30)
    
    print(f"{'Total CAGR':<15} | {s1['CAGR']:.2%}          | {s2['CAGR']:.2%}          | {diff_cagr:+.2%}")
    print(f"{'Total MDD':<15} | {s1['MDD']:.2%}          | {s2['MDD']:.2%}          | {diff_mdd:+.2%}")
    
    # Period 5
    p5_ret1 = p1['Final_Equity']/p1['Final_Equity'] # wait logic error in calc
    # Recalculate Period 5 Return
    p5_ret_old = (p1['Final_Equity'] / (p1['Final_Equity'] / (1+0))) # No.. logic in struct
    # Let's trust print output for manual inspection, but for diff here:
    
    # We need stored values
    # Hacky:
    pass

if __name__ == "__main__":
    main()
