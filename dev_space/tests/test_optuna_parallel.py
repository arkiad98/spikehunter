import optuna
import os
import time
import threading

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    # Simulate CPU work
    sum_val = 0
    for i in range(1000000):
        sum_val += i * x
    
    print(f"Trial {trial.number}: PID={os.getpid()}, Thread={threading.get_ident()}")
    return (x - 2) ** 2

if __name__ == "__main__":
    print(f"Main PID: {os.getpid()}")
    study = optuna.create_study(direction="minimize")
    print("Starting optimization with n_jobs=2...")
    study.optimize(objective, n_trials=4, n_jobs=2)
