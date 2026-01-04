
import os
import yaml
import sys
import subprocess
from datetime import datetime

def run_experiment(lookback_years: int):
    print(f"\n" + "="*60)
    print(f" EXPERIMENT START: Lookback {lookback_years} Years ")
    print("="*60)
    
    # 1. Create/Modify Config
    # We will invoke run_pipeline.py with a specific settings file?
    # run_pipeline currently uses 'config/settings.yaml' by default.
    # We need to temporarily swap file or modify logic.
    # Safer: Create a new runner script or temporarily overwrite settings.yaml (with backup)
    # But user asked for "Separate workspace". Since we are in same dir, unique output dir is key.
    
    # Let's write a temporary settings file and pass it if supported, 
    # OR modify the existing one and revert.
    # Current codebase might hardcode 'config/settings.yaml'. 
    # Let's assume we modify 'config/settings.yaml' but backup first.
    
    # Actually, we can use the param override feature if we call modules directly.
    # But calling `run_advanced_optimization_framework` is complex.
    # Let's subprocess `python run_pipeline.py` but we need to change inputs.
    
    # Strategy:
    # 1. Backup settings.yaml
    # 2. Modify settings.yaml (Train Period)
    # 3. Run Pipeline (Menu 2 -> 3) - AUTOMATED
    # 4. Restore settings.yaml
    
    # But automating menu input in subprocess is tricky.
    # Better: Import the WFO function directly.
    
    # from modules.optimization_ml import run_advanced_optimization_framework # [Removed]
    from modules.utils_io import read_yaml
    
    # Load default settings
    cfg_path = "config/settings.yaml"
    cfg = read_yaml(cfg_path)
    
    # Modify Lookback
    # settings.yaml -> machine_learning -> classification_train_months ??
    # OR is it hardcoded in WFO logic?
    # WFO logic usually takes (train_period_months) argument.
    
    # Let's inspect `modules/optimization_ml.py` to see how it determines window.
    # Assuming it reads from `ml_params` or `optimization` config.
    
    # For this experiment, let's inject the parameter.
    # We'll create a modified runner here.
    
    print(f">> Configuring for {lookback_years} years ({lookback_years*12} months)...")
    
    # Modify Lookback (WFO train window)
    if 'walk_forward' not in cfg:
        cfg['walk_forward'] = {}
        
    cfg['walk_forward']['train_months'] = int(lookback_years * 12)
    # Ensure test_months is 6 (default)
    if 'test_months' not in cfg['walk_forward']:
        cfg['walk_forward']['test_months'] = 6
        
    print(f">> Configured WFO: Train {cfg['walk_forward']['train_months']}m / Test {cfg['walk_forward']['test_months']}m")
    
    # Unique Output Dir
    exp_name = f"EXP_Lookback_{lookback_years}Y_{datetime.now().strftime('%H%M%S')}"
    # We can't easily force output dir without code change, but WFO creates timestamped folder.
    # We'll detect the newest folder.
    
    # Save temp config
    temp_cfg_path = f"config/settings_temp_{lookback_years}y.yaml"
    with open(temp_cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
        
    print(f">> Settings saved to {temp_cfg_path}")
    
    # Execute WFO
    # We need to call `run_advanced_optimization_framework` with this config.
    # But `run_advanced_optimization_framework` usually reads `config/settings.yaml`.
    # Quick Fix: Rename temp to settings.yaml (Backup first!)
    
    backup_path = "config/settings.yaml.bak_exp"
    if not os.path.exists(backup_path):
        os.rename("config/settings.yaml", backup_path)
        
    # Overwrite
    with open("config/settings.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
        
    try:
        # Import and Run Direct Function
        # We need to add project root to sys.path if not present
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
            
        from run_pipeline import run_walk_forward_pipeline
        
        # Strategy name resolution
        strat_name = list(cfg.get('strategies', {}).keys())[0] if cfg.get('strategies') else "SpikeHunter"
        
        print(f">> Starting WFO for strategy: {strat_name}")
        
        # Verify
        if os.path.exists("config/settings.yaml"):
             print(">> temporary settings.yaml exists.")
             # Debug content
             # print(read_yaml("config/settings.yaml").keys())
        else:
             print(">> Error: temporary settings.yaml MISSING.")
             
        run_walk_forward_pipeline("config/settings.yaml", strat_name)
        
    except Exception as e:
        print(f"Error during WFO execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore
        if os.path.exists(backup_path):
            if os.path.exists("config/settings.yaml"):
                os.remove("config/settings.yaml")
            os.rename(backup_path, "config/settings.yaml")
            print(">> Settings restored.")

def main():
    # Run 2 Years
    # run_experiment(2)
    
    # Run 3 Years
    # run_experiment(3)
    
    print("Use command line args: python experiment.py [2|3]")
    if len(sys.argv) > 1:
        run_experiment(int(sys.argv[1]))

if __name__ == "__main__":
    main()
