from modules.backtest import run_backtest
from modules.utils_io import read_yaml
import os

def main():
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    
    # Strategy name from settings
    strategy_name = "SpikeHunter_R1_BullStable"
    
    print(f"Running verification backtest for {strategy_name}...")
    
    # Run backtest for 2024-2025
    result = run_backtest(
        run_dir="data/proc/backtest/verification_run",
        strategy_name=strategy_name,
        settings_path=settings_path,
        start="2025-01-01",
        end="2025-06-30",
        quiet=False
    )
    
    if result:
        metrics = result['metrics']
        print("\n[Verification Result]")
        print(f"CAGR: {metrics['CAGR_raw']*100:.2f}%")
        print(f"MDD: {metrics['MDD_raw']*100:.2f}%")
        print(f"Win Rate: {metrics['win_rate_raw']*100:.2f}%")
        print(f"Total Trades: {metrics['총거래횟수']}")
    else:
        print("Backtest failed to produce results.")

if __name__ == "__main__":
    main()
