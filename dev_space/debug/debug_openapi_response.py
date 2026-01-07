
import sys
import os
import requests
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from modules.utils_io import read_yaml

def debug_response():
    settings_path = os.path.join(os.path.dirname(__file__), '../../config/settings.yaml')
    cfg = read_yaml(settings_path)
    api_key = cfg.get('ml_params', {}).get('krx_api_key')
    
    url = "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"
    headers = {"User-Agent": "Mozilla/5.0", "AUTH_KEY": api_key}
    params = {"basDd": "20260102"}
    
    print(f"Requesting {url} with params {params}")
    resp = requests.get(url, params=params, headers=headers)
    print(f"Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print("Raw Data (First item in OutBlock1):")
        if isinstance(data, dict):
             for k, v in data.items():
                 if isinstance(v, list) and len(v) > 0:
                     print(f"Key: {k}")
                     print(v[0]) # Print first item
                     
                     # Print ALL items to find the one with prices
                     for i, item in enumerate(v):
                         print(f"\n[Item {i}]")
                         print(item)
    else:
        print(resp.text)

if __name__ == "__main__":
    debug_response()
