import json
import pandas as pd
import re
import os

LOG1 = "logs/20251231_203532_pipeline.json" # Light (15 features)
LOG2 = "logs/20260101_000543_pipeline.json" # Heavy (35 features)

def parse_log(filepath):
    metrics = {
        "AP": [],
        "CAGR": None,
        "MDD": None,
        "WinRate": None,
        "Trades": None
    }
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return metrics

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                msg = record.get('message', '')
                
                # Extract AP
                if "Ensemble AP:" in msg:
                    # [Fold 1] Ensemble AP: 0.1807 (L=0.1807)
                    match = re.search(r"Ensemble AP: ([\d\.]+)", msg)
                    if match:
                        metrics["AP"].append(float(match.group(1)))
                        
                # Extract Backtest Metrics (This depends on how the log is structured)
                # Looking for "CAGR", "MDD" in the message
                if "CAGR" in msg and "%" in msg:
                    # Example: CAGR: 1,068.32%
                    match = re.search(r"CAGR.*?([\d,\.]+)%", msg)
                    if match:
                        metrics["CAGR"] = match.group(1)
                
                if "MDD" in msg and "%" in msg:
                    match = re.search(r"MDD.*?([\-\d,\.]+)%", msg)
                    if match:
                        metrics["MDD"] = match.group(1)
                        
                if "승률" in msg or "Win Rate" in msg:
                     match = re.search(r"(?:승률|Win Rate).*?([\d\.]+)%", msg)
                     if match:
                         metrics["WinRate"] = match.group(1)

                if "총 거래수" in msg or "Total Trades" in msg:
                     match = re.search(r"(?:총 거래수|Total Trades).*?(\d+)", msg)
                     if match:
                         metrics["Trades"] = match.group(1)

            except json.JSONDecodeError:
                continue
                
    return metrics

m1 = parse_log(LOG1)
m2 = parse_log(LOG2)

print(f"{'Metric':<20} | {'Old (15 Feat)':<15} | {'New (35 Feat)':<15}")
print("-" * 56)
avg_ap1 = sum(m1['AP'])/len(m1['AP']) if m1['AP'] else 0
avg_ap2 = sum(m2['AP'])/len(m2['AP']) if m2['AP'] else 0
print(f"{'Average AP':<20} | {avg_ap1:.4f}{' '*9} | {avg_ap2:.4f}")
print(f"{'CAGR':<20} | {str(m1.get('CAGR', 'N/A')):<15} | {str(m2.get('CAGR', 'N/A')):<15}")
print(f"{'MDD':<20} | {str(m1.get('MDD', 'N/A')):<15} | {str(m2.get('MDD', 'N/A')):<15}")
print(f"{'Win Rate':<20} | {str(m1.get('WinRate', 'N/A')):<15} | {str(m2.get('WinRate', 'N/A')):<15}")
print(f"{'Trades':<20} | {str(m1.get('Trades', 'N/A')):<15} | {str(m2.get('Trades', 'N/A')):<15}")
