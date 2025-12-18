import os
from datetime import datetime
from modules.backtest import run_backtest
from modules.utils_io import read_yaml

def main():
    settings_path = "config/settings.yaml"
    cfg = read_yaml(settings_path)
    
    strategy_name = "SpikeHunter_R1_BullStable" # Use the configured strategy
    
    # Run for 2024
    start_date = "2024-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/proc/backtest/run_final_{run_id}"
    
    print(f"Starting Final Backtest for {strategy_name} ({start_date} ~ {end_date})...")
    
    result = run_backtest(
        run_dir=run_dir,
        strategy_name=strategy_name,
        settings_path=settings_path,
        start=start_date,
        end=end_date,
        save_to_db=True
    )
    
    if result:
        metrics = result['metrics']
        print("\n" + "="*60)
        print(f"   [Final Backtest Result 2024]")
        print(f"   - CAGR: {metrics['CAGR_raw']*100:.2f}%")
        print(f"   - MDD: {metrics['MDD_raw']*100:.2f}%")
        print(f"   - Win Rate: {metrics['win_rate_raw']*100:.2f}% ({metrics['총거래횟수']} trades)")
        print(f"   - Sharpe: {metrics['Sharpe_raw']:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()
