import json
import re
import pandas as pd
import numpy as np
import sys
import os

def parse_log_file(file_path, label):
    trials = []
    print(f"Parsing: {file_path} as {label}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                message = log_entry.get('message', '')
                
                if ' >> [Finished] Trial' in message:
                    score_match = re.search(r"Score=([\d\.\-]+)", message)
                    score = float(score_match.group(1)) if score_match else -np.inf
                    
                    params_match = re.search(r"Params=(\{.*?\})", message)
                    if params_match:
                        params_str = params_match.group(1).replace("'", '"')
                        try:
                            params = json.loads(params_str)
                            trial_data = {'score': score, 'batch': label}
                            trial_data.update(params)
                            trials.append(trial_data)
                        except Exception as e:
                            print(f"Failed to parse params json: {params_str} - {e}")
            except Exception as e:
                pass
                
    return trials

def analyze_convergence(df, batch_name="All"):
    if df.empty:
        print(f"[{batch_name}] No trial data found.")
        return

    print(f"\n{'='*20} Analysis: {batch_name} {'='*20}")
    print(f"Total Trials: {len(df)}")
    if 'score' in df:
        print(f"Best Score: {df['score'].max():.4f}")
    
    df_sorted = df.sort_values(by='score', ascending=False)
    
    # Calculate top 10% count
    top_10_percent_count = int(len(df) * 0.1)
    
    top_counts = [10, top_10_percent_count]
    
    for n in top_counts:
        if n > len(df) or n == 0:
            continue
            
        top_df = df_sorted.head(n)
        label_text = f"Top {n} Trials" if n == 10 else f"Top 10% ({n}) Trials"
        
        print(f"\n--- {label_text} ---")
        print(f"Score Mean: {top_df['score'].mean():.4f} | Std: {top_df['score'].std():.4f}")
        
        # summary = top_df.describe().T
        # print(summary[['mean', 'std', 'min', 'max']])
        
        for col in top_df.columns:
            if col in ['score', 'batch']: continue
            if not pd.api.types.is_numeric_dtype(top_df[col]): continue

            std = top_df[col].std()
            mean = top_df[col].mean()
            cv = std / abs(mean) if mean != 0 else 0
            
            status = "CONVERGED" if cv < 0.05 else ("STABLE" if cv < 0.15 else "UNSTABLE")
            print(f"  > {col:<15}: Mean={mean:.4f}, Std={std:.4f}, CV={cv:.4f} [{status}]")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_log_convergence.py <log_file_path1> [log_file_path2 ...]")
        return
        
    all_trials = []
    
    # 3 logs expected usually
    # Labeling logic: just verify basic mapping
    # 20260109_131732 -> Batch_1 (Initial)
    # 20260109_134614 -> Batch_2 (Combined 4000)
    # 20260109_142050 -> Batch_3 (High ML)
    
    log_files = sys.argv[1:]
    
    for i, log_file in enumerate(log_files):
        label = f"Batch_{i+1}"
        trials = parse_log_file(log_file, label)
        all_trials.extend(trials)
        
        # Individual Analysis
        df_batch = pd.DataFrame(trials)
        analyze_convergence(df_batch, batch_name=os.path.basename(log_file))

    # Combined Analysis
    df_all = pd.DataFrame(all_trials)
    analyze_convergence(df_all, batch_name="Combined All (6000)")

if __name__ == "__main__":
    main()
