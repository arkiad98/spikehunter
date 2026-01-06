

import json
import os
import re

log_path = r"d:\spikehunter\logs\20251230_090210_pipeline.json"

def analyze_wf_log(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Analyzing log: {path}")
    
    # Try reading with different encodings
    content = ""
    for enc in ['utf-8', 'cp949', 'euc-kr']:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            print(f"Successfully read with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        print("Failed to read file with standard encodings.")
        return

    # Try parsing as JSON first
    try:
        data = json.loads(content)
        entries = data if isinstance(data, list) else data.get('logs', [])
    except:
        # Fallback: regex for JSON objects if it's a stream of objects
        print("Raw text parsing...")
        entries = []
        # specific logic for the user's log format (assuming standard python logger json)
        # But if it's failed, let's just look at line by line for keywords
        lines = content.splitlines()
        
        current_period = None
        period_data = []
        
        for line in lines:
            if "Walk-Forward Period" in line and "Start" in line:
                if current_period: period_data.append(current_period)
                current_period = {"info": line.strip(), "params": [], "metrics": []}
                
            if current_period:
                if "최적 파라미터:" in line:
                    current_period["params"].append(line.strip())
                if "백테스트 결과" in line:
                    current_period["metrics"].append("--- Result ---")
                if "최종 자산:" in line or "수익률(CAGR):" in line or "MDD:" in line or "승률:" in line:
                    current_period["metrics"].append(line.strip())
                    
        if current_period: period_data.append(current_period)
        
        # Display
        for i, p in enumerate(period_data):
            print(f"\n[Period {i+1}]")
            print(p['info'])
            for param in p['params']:
                print(f"  - {param}")
            for m in p['metrics']:
                print(f"  > {m}")
        return

    # If JSON parsing worked
    print(f"Parsed {len(entries)} log entries.")
    
    current_period = {}
    periods = []
    
    for entry in entries:
        msg = entry.get('message', '')
        
        if "Walk-Forward Period" in msg and "Start" in msg:
            if current_period: periods.append(current_period)
            current_period = {'info': msg, 'params': [], 'metrics': []}
            
        if "최적 파라미터:" in msg:
            current_period.setdefault('params', []).append(msg)
            
        if "최종 자산:" in msg or "수익률(CAGR):" in msg or "MDD:" in msg or "승률:" in msg:
            current_period.setdefault('metrics', []).append(msg)
            
    if current_period: periods.append(current_period)
    
    for i, p in enumerate(periods):
        print(f"\n[Period {i+1}]")
        print(p.get('info', ''))
        for param in p.get('params', []):
            print(f"  - Opt: {param}")
        for m in p.get('metrics', []):
            print(f"  > {m}")

