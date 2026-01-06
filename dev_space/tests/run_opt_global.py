
from modules.optimization_ml import run_ml_optimization

if __name__ == "__main__":
    print(">> Starting Time-Lagged Global Model Optimization (5 Trials)...")
    # Using 5 trials for speed in interactive mode
    run_ml_optimization("config/settings.yaml", n_trials=5)
