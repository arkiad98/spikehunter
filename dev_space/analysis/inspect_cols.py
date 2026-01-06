
import pandas as pd

wf_path = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260102_130851\wf_tradelog.csv"
std_path = r"D:\spikehunter\data\proc\backtest\run_20260102_133019\tradelog.parquet"

def inspect_columns(path, name, is_parquet=False):
    print(f"--- Columns for {name} ---")
    try:
        if is_parquet:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        print(df.columns.tolist())
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {name}: {e}")
    print("\n")

inspect_columns(wf_path, "Walk-Forward", is_parquet=False)
inspect_columns(std_path, "Standard Run", is_parquet=True)
