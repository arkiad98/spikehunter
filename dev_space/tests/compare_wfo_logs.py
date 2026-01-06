import json
import re

def parse_log_for_wfo(log_path):
    wfo_results = {}
    current_period = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Assuming JSON log format. If plain text, regex is needed.
                if line.strip().startswith('{'):
                    data = json.loads(line)
                    msg = data.get('message', '')
                    
                    # Detect Period Start or Optimization Result
                    # Pattern: " >> [WFO] 전략 최적화 시작 (2020-01-01 ~ 2022-01-01)"
                    if "전략 최적화 시작" in msg:
                        match = re.search(r'\((.*?)\)', msg)
                        if match:
                            current_period = match.group(1)
                            wfo_results[current_period] = {'best_score': None, 'best_params': None}
                            
                    # Pattern: " >> [WFO] 최적화 완료. Best Score: 0.1234"
                    if "최적화 완료. Best Score:" in msg and current_period:
                        score = float(msg.split("Best Score:")[1].strip())
                        wfo_results[current_period]['best_score'] = score
                        
            except Exception as e:
                pass
    return wfo_results

def compare_logs(path1, path2):
    print(f"Comparing:")
    print(f"1: {path1}")
    print(f"2: {path2}")
    
    res1 = parse_log_for_wfo(path1)
    res2 = parse_log_for_wfo(path2)
    
    print("\n[Comparison Result]")
    all_periods = sorted(list(set(res1.keys()) | set(res2.keys())))
    
    for p in all_periods:
        s1 = res1.get(p, {}).get('best_score', 'N/A')
        s2 = res2.get(p, {}).get('best_score', 'N/A')
        print(f"Period {p}:")
        print(f"  - Baseline: {s1}")
        print(f"  - Current : {s2}")
        
        if isinstance(s1, float) and isinstance(s2, float):
            diff = s2 - s1
            print(f"  - Diff    : {diff:.4f}")

if __name__ == "__main__":
    log1 = r"D:\spikehunter\logs\20260102_130231_pipeline.json"
    log2 = r"D:\spikehunter\logs\20260102_155823_pipeline.json"
    compare_logs(log1, log2)
