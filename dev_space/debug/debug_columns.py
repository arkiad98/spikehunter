import pandas as pd
import numpy as np

path_wf = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260108_150918\period_8\result\tradelog.parquet"

print("Loading WF (Period 8)...")
try:
    df_wf = pd.read_parquet(path_wf)
    print("Columns:", df_wf.columns.tolist())
    print("First row:", df_wf.head(1).to_dict())
except Exception as e:
    print(f"Error loading WF: {e}")
