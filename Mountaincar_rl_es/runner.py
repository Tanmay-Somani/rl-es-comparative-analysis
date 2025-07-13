# runner.py
import multiprocessing as mp
from rl_non_fl import run_rl_non_fl
from es_distributed import run_es_distributed
from fl_rl import run_fl_rl
from utils import plot_results

if __name__ == "__main__":
    # This is crucial for multiprocessing on some platforms (macOS, Windows)
    mp.freeze_support()

    # Initialize the dictionary to store all benchmark data
    log_data = {
        'rl_non_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': []},
        'es_distributed': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': []},
        'rl_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': [], 'communication_overhead': []},
    }

    # --- Run Benchmarks Sequentially ---
    log_data = run_rl_non_fl(log_data)
    log_data = run_es_distributed(log_data)
    log_data = run_fl_rl(log_data)

    # --- Generate and Save Plots ---
    plot_results(log_data)