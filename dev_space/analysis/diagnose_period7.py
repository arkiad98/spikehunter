
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import joblib

# Add project root to path
sys.path.append(os.path.abspath("d:/spikehunter"))

from modules.utils_io import read_yaml
from modules.utils_logger import logger


def diagnose_period7():
    print(">>> Starting Period 7 Diagnosis (2025-01-01 ~ 2025-06-30)")
    
    # 1. Load Settings & Params
    settings = read_yaml("d:/spikehunter/config/settings.yaml")
    
    # Params from Period 7 Log (Step 620)
    params = {
        'target_r': 0.092749, 
        'stop_r': -0.048393, 
        'min_ml_score': 0.280228, 
        'max_market_vol': 0.031123, 
        'vbo_k': 0.663105, 
        'min_mfi': 49.5972,
        'target_hold_period': 5 # From settings default
    }
    
    # 2. Load Data (Dataset)
    # Using the path found in recent logs
    dataset_path = "d:/spikehunter/data/proc/ml_dataset/ml_classification_dataset.parquet"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for Period 7
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2025-06-30")
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    print(f"Data loaded: {len(df)} rows ({df['date'].min().date()} ~ {df['date'].max().date()})")
    
    # 3. Analyze Market Features
    print("\n[Market Analysis]")
    if 'market_bullish' in df.columns:
        bull_days = df.groupby('date')['market_bullish'].max()
        print(f"Total Days: {len(bull_days)}")
        print(f"Bullish Days: {bull_days.sum()} ({bull_days.mean()*100:.1f}%)")
        print(f"Bearish Days: {len(bull_days) - bull_days.sum()}")
        
        # Check consecutive bearish days
        # Analyze why user thinks it's bullish? 
        # Maybe KOSPI was rising but market_bullish (ma200) was 0?
        # Let's load KOSPI index to verify
        try:
            kospi_df = pd.read_parquet("d:/spikehunter/data/raw/index/kospi.parquet") # Adjust path if needed or use load_index_data
            kospi_df['date'] = pd.to_datetime(kospi_df['date'])
            k_period = kospi_df[(kospi_df['date'] >= start_date) & (kospi_df['date'] <= end_date)]
            if not k_period.empty:
                k_start = k_period.iloc[0]['close']
                k_end = k_period.iloc[-1]['close']
                print(f"KOSPI Return: {k_start:.2f} -> {k_end:.2f} ({(k_end/k_start - 1)*100:.2f}%)")
        except:
            print("Could not load KOSPI raw data for verification.")
            
    else:
        print("Feature 'market_bullish' NOT FOUND.")

    # 4. Simulate Strategy (Simplified)
    print("\n[Strategy Simulation]")
    
    # Load Model (if needed for score) or check if 'ml_score' exists?
    # Usually dataset doesn't have ml_score pre-calculated unless saved.
    # WFO saves predictions?
    # For diagnosis, I will assume we need to predict or load predictions.
    # Checking for predictions_latest.csv or similar.
    # But to be precise, let's load the model and predict.
    
    model_path = "d:/spikehunter/data/models/lgbm_model.joblib" # Default path
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        # Prepare Features
        feature_cols = [c for c in df.columns if c not in ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'target', 'target_log_ret', 'label_class', 'year', 'month', 'day', 'weekday', 'fwd_max_ret', 'fwd_ret', 'inst_net_val', 'foreign_net_val', 'value', 'change', 'market_index', 'regime', 'kospi_close', 'kospi_value']]
        # Use columns from model
        if hasattr(model, 'feature_name_'):
             feature_cols = model.feature_name_
             
        X = df[feature_cols].fillna(0)
        print("Predicting ML Scores...")
        df['ml_score'] = model.predict_proba(X)[:, 1]
    else:
        print("Model not found. Cannot simulate trades accurately without ML scores.")
        return

    # Simulation Loop
    portfolio = {} 
    trades = []
    cash = 10_000_000
    fee = 0.0015
    
    dates = sorted(df['date'].unique())
    
    for date in tqdm(dates):
        today = df[df['date'] == date].set_index('code')
        
        # Sell
        sold = []
        for code, pos in portfolio.items():
            if code not in today.index: continue
            row = today.loc[code]
            
            # Exit Logic (Simplified)
            open_p, high, low, close = row['open'], row['high'], row['low'], row['close']
            entry_p = pos['entry_price']
            
            target_p = entry_p * (1 + params['target_r'])
            stop_p = entry_p * (1 + params['stop_r']) # stop_r is negative
            
            exit_p = None
            reason = ""
            
            if open_p >= target_p: exit_p = open_p; reason = "GapTP"
            elif open_p <= stop_p: exit_p = open_p; reason = "GapSL"
            elif high >= target_p: exit_p = target_p; reason = "TP"
            elif low <= stop_p: exit_p = stop_p; reason = "SL"
            elif (date - pos['entry_date']).days >= params['target_hold_period']:
                exit_p = close; reason = "Time"
                
            if exit_p:
                ret = (exit_p - entry_p) / entry_p - fee
                # Actually calculating return
                ret = (exit_p - entry_p) / entry_p - fee
                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'code': code,
                    'return': ret,
                    'reason': reason,
                    'entry_score': pos['score'],
                    'market_bullish_entry': pos['market_bullish']
                })
                sold.append(code)
        
        for c in sold: del portfolio[c]
        
        # Buy
        # Bullish Filter? Strategy checking 'max_market_vol' (low volatility preferred?)
        # And min_mfi, etc.
        # But wait, usually 'market_bullish' feature is used as a weight or filter in simple strategies?
        # In SpikeHunter_R1_BullStable:
        # It typically doesn't filter by market_bullish unless hardcoded.
        # It DOES filter by Volatility: if market_volatility > params['max_market_vol']: skip
        
        # Get market volatility for today
        # market_vol is same for all stocks on same day
        m_vol = today['market_volatility'].iloc[0] if 'market_volatility' in today.columns else 0
        if m_vol > params['max_market_vol']:
            continue # Skip buying if volatility is too high
            
        candidates = today[today['ml_score'] >= params['min_ml_score']]
        # Additional filters
        if 'mfi_14' in today.columns:
            candidates = candidates[candidates['mfi_14'] >= params['min_mfi']]
        if 'vbo_k' in today.columns:
            pass # VBO K Logic usually inside labeling or specific pattern check. 
                 # Assuming VBO feature exists or logic handles it.
                 # For diagnosis, let's treat it as minor if not readily available as feature.
                 
        candidates = candidates.sort_values('ml_score', ascending=False)
        
        for code, row in candidates.iterrows():
            if len(portfolio) >= 5: break
            if code in portfolio: continue
            
            portfolio[code] = {
                'entry_price': row['close'],
                'entry_date': date,
                'shares': 1, # Dummy
                'score': row['ml_score'],
                'market_bullish': row.get('market_bullish', 0)
            }

    # Summary
    if not trades:
        print("No trades executed.")
    else:
        trades_df = pd.DataFrame(trades)
        print(f"\n[Trade Summary]")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Win Rate: {(trades_df['return'] > 0).mean()*100:.1f}%")
        print(f"Avg Return: {trades_df['return'].mean()*100:.2f}%")
        
        print("\n[Exit Reason Breakdown]")
        print(trades_df['reason'].value_counts())
        
        # Analyze performance by Reason
        print("\n[Performance by Reason]")
        print(trades_df.groupby('reason')['return'].describe()[['count', 'mean', 'min', 'max']])

        print("\n[Losing Trades Analysis]")
        losers = trades_df[trades_df['return'] < 0]
        if not losers.empty:
            print(f"Count: {len(losers)}")
            print(f"Avg Loss: {losers['return'].mean()*100:.2f}%")
            
            # Check Volatility for Losers vs Winners
            # We need to link trade entry date to market_volatility
            # Assuming 'market_bullish_entry' is a proxy, but let's see if we captured vol.
            # We didn't capture 'market_vol_entry' in the trade dict. Let's add it in next run if needed.
            # But we can infer from date if we had the market df.
            # For now, let's look at the 'reason' of losers specifically.
            print("Loser Reasons:")
            print(losers['reason'].value_counts())
            
            print("\nTop 5 Worst Trades:")
            print(losers.sort_values('return').head(5)[['entry_date', 'exit_date', 'code', 'return', 'reason', 'entry_score']])

    # Volatility Impact Check
    # Need to reload market features to map date -> volatility
    # kospi_df was loaded in step 3? No, we skipped it or it failed.
    # Let's try to load dataset again or features to get vol map.
    vol_map = df.set_index('date')['market_volatility'].to_dict() # Assuming it's in df (it should be allowed in step 3)
    
    if trades:
        trades_df['market_vol'] = trades_df['entry_date'].map(vol_map)
        print("\n[Volatility Impact Analysis]")
        print(f"Avg Vol (All Trades): {trades_df['market_vol'].mean():.4f}")
        print(f"Avg Vol (Winners): {trades_df[trades_df['return']>0]['market_vol'].mean():.4f}")
        print(f"Avg Vol (Losers): {trades_df[trades_df['return']<0]['market_vol'].mean():.4f}")
        
        # Segmented Performance
        high_vol = trades_df[trades_df['market_vol'] > 0.015] # benchmark
        low_vol = trades_df[trades_df['market_vol'] <= 0.015]
        
        print(f"\nHigh Volatility (>0.015) Trades: {len(high_vol)}")
        if not high_vol.empty:
            print(f"Win Rate: {(high_vol['return']>0).mean()*100:.1f}%")
            print(f"Avg Return: {high_vol['return'].mean()*100:.2f}%")
            
        print(f"\nLow Volatility (<=0.015) Trades: {len(low_vol)}")
        if not low_vol.empty:
            print(f"Win Rate: {(low_vol['return']>0).mean()*100:.1f}%")
            print(f"Avg Return: {low_vol['return'].mean()*100:.2f}%")


if __name__ == "__main__":
    diagnose_period7()
