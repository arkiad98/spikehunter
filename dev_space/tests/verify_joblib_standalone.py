import time
import os
from joblib import Parallel, delayed

def heavy_computation(x):
    start = time.time()
    # CPU-bound task: matrix multiplication simulation or simple loop
    result = 0
    for i in range(5000000):
        result += i * i * x
    return time.time() - start

def verify_standalone_parallelism():
    print(f"Main Process PID: {os.getpid()}")
    tasks = [1, 2, 3, 4]
    
    # 1. Sequential Execution
    print("\n[Test 1] Sequential Execution (n_jobs=1)")
    start_seq = time.time()
    results_seq = [heavy_computation(x) for x in tasks]
    duration_seq = time.time() - start_seq
    print(f"Total Duration: {duration_seq:.4f}s")
    print(f"Avg Task Duration: {sum(results_seq)/len(results_seq):.4f}s")

    # 2. Parallel Execution
    print("\n[Test 2] Parallel Execution (n_jobs=4)")
    start_par = time.time()
    results_par = Parallel(n_jobs=4)(delayed(heavy_computation)(x) for x in tasks)
    duration_par = time.time() - start_par
    print(f"Total Duration: {duration_par:.4f}s")
    
    # Validation
    speedup = duration_seq / duration_par
    print(f"\n[Result] Speedup: {speedup:.2f}x")
    
    if speedup > 2.0:
        print("SUCCESS: Multiprocessing is working effectively.")
    else:
        print("WARNING: Speedup is lower than expected. (Check CPU/Cores)")

if __name__ == "__main__":
    verify_standalone_parallelism()
