import sys
import os
sys.path.append(os.getcwd())
from modules.backtest import run_backtest

print("Starting Validation Backtest...")
try:
    result = run_backtest(
        run_dir="data/proc/backtest/validation_run",
        strategy_name="SpikeHunter_R1_BullStable",
        settings_path="config/settings.yaml",
        save_to_db=False,
        quiet=False
    )
    if result and not result['daily_equity'].empty:
        cagr = result['metrics']['CAGR_raw']
        mdd = result['metrics']['MDD_raw']
        wr = result['metrics']['win_rate_raw']
        print(f"Validation Result: CAGR={cagr:.2%}, MDD={mdd:.2%}, WinRate={wr:.2%}")
    else:
        print("Validation Failed: No result returned.")
except Exception as e:
    print(f"Error: {e}")
