import sys
import os
sys.path.append("d:/spikehunter")
from modules.backtest import run_backtest
from modules.utils_io import read_yaml

def verify():
    print(">> Verifying Deployed SpikeHunter v4.0...")
    
    settings_path = "d:/spikehunter/config/settings.yaml"
    if not os.path.exists(settings_path):
        print("Settings not found!")
        return

    # Run Backtest
    run_dir = "d:/spikehunter/data/proc/backtest/verify_v4"
    strategy_name = "SpikeHunter_v4"
    
    # Override date to ensure full period check
    result = run_backtest(
        run_dir=run_dir,
        strategy_name=strategy_name,
        settings_path=settings_path,
        start="2020-01-01",
        end="2025-12-31",
        quiet=False
    )
    
    if result:
        metrics = result['metrics']
        print("\n[Verification Result]")
        print(f"CAGR: {metrics['CAGR_raw']*100:.2f}%")
        print(f"Final Equity: {result['daily_equity']['equity'].iloc[-1]:,.0f}")
        
        if metrics['CAGR_raw'] > 5.0: # Expecting > 500%
            print(">> SUCCESS: Deployment Verified.")
        else:
            print(">> FAILURE: Performance mismatch.")
    else:
        print(">> FAILURE: Backtest returned None.")

if __name__ == "__main__":
    verify()
