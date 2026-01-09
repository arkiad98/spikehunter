
from modules import collect
import os

def run():
    settings_path = "config/settings.yaml"
    start_date = "2025-09-01"
    end_date = "2025-12-31"
    
    print(f"Collecting data for {start_date} ~ {end_date}...")
    collect.run_collect(settings_path, start=start_date, end=end_date, use_meta=False)
    print("Collection Complete.")

if __name__ == "__main__":
    run()
