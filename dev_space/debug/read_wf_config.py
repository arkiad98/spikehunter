
import yaml

wf_config_path = r"D:\spikehunter\data\proc\backtest\WF-SpikeHunter_R1_BullStable-20260102_130851\temp_settings.yaml"

try:
    with open(wf_config_path, 'r', encoding='utf-8') as f:
         config = yaml.safe_load(f)
         print("--- WF Strategies Config ---")
         if 'strategies' in config:
             print(config['strategies'])
         else:
             print("No 'strategies' section found.")
             
         print("\n--- WF Optimization Config ---")
         if 'optimization' in config:
             print(config['optimization'])
except Exception as e:
    print(f"Error: {e}")
