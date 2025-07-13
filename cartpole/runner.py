# runner.py
import multiprocessing as mp
from rl_non_fl import run_rl_non_fl
from es_distributed import run_es_distributed
from fl_rl import run_fl_rl
from fl_es import run_fl_es
from utils import plot_results

if __name__ == "__main__":
    # This is crucial for multiprocessing on some platforms (macOS, Windows)
    mp.freeze_support()

    # Initialize the dictionary to store all benchmark data
    log_data = {
        'rl_non_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': []},
        'es_distributed': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': []},
        'rl_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': [], 'communication_overhead': []},
        'es_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': [], 'communication_overhead': []},
    }

    print("Starting CartPole RL/ES Comparative Analysis")
    print("=" * 50)

    # --- Run Benchmarks Sequentially ---
    print("\n1. Running Non-Federated Reinforcement Learning...")
    log_data = run_rl_non_fl(log_data)
    
    print("\n2. Running Distributed Evolutionary Strategies...")
    log_data = run_es_distributed(log_data)
    
    print("\n3. Running Federated Reinforcement Learning...")
    log_data = run_fl_rl(log_data)
    
    print("\n4. Running Federated Evolutionary Strategies...")
    log_data = run_fl_es(log_data)

    # --- Generate and Save Plots ---
    print("\n5. Generating comparison plots...")
    plot_results(log_data)
    
    print("\nCartPole RL/ES Comparative Analysis Complete!")
    print("All results and plots have been saved.") 