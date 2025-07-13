# main_runner.py
import argparse
import subprocess
import os
import time
import pickle
import numpy as np
from argparse import RawTextHelpFormatter
from grid_headless import GridEnvironment

# A list of experiments for the parser's choices
EXPERIMENTS = ["centralized_rl", "federated_rl", "centralized_es", "federated_es"]

def run_script(script_name, render=False, runs=1):
    """Executes a training script, can handle multiple runs for robustness testing."""
    experiment_name = script_name.replace('.py', '')
    
    for i in range(runs):
        print(f"\n>>> Starting Run {i+1}/{runs} for experiment: {experiment_name}...")
        
        log_dir = os.path.join("logs", experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file_name = f"training_log_run_{i+1}.csv" if runs > 1 else "training_log.csv"
        log_file_path = os.path.join(log_dir, log_file_name)
        
        command = ["python", script_name, "--log_file", log_file_path]
        if render and runs == 1:
            command.append("--render")
            
        try:
            process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
            print(process.stdout)
            if process.stderr: print(f"--- Script Errors ---\n{process.stderr}")
            print(f">>> Run {i+1}/{runs} completed successfully.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"!!! An error occurred during run {i+1}/{runs} of {script_name}.")
            print(e.stdout if hasattr(e, 'stdout') else "No stdout.")
            print(e.stderr if hasattr(e, 'stderr') else "No stderr.")
            break
            
    if runs == 1:
        print("\n>>> Calling analysis plotter for a single report...")
        plotter_command = ["python", "analysis_plotter.py", "single", "--log_file", log_file_path, "--name", experiment_name]
        subprocess.run(plotter_command)
    else:
        print(f"\n>>> All {runs} runs complete. Use 'python analysis_plotter.py compare' to see robustness results.")

def demonstrate_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    with open(model_path, 'rb') as f: data = pickle.load(f)
    model, env_config = data['model'], data['env_config']
    
    env = GridEnvironment()
    env.set_config(env_config)
    state = env.reset()
    done = False
    
    is_rl_model = isinstance(model, np.ndarray)
    model_type = "Reinforcement Learning (Q-table)" if is_rl_model else "Evolutionary Strategy (Policy)"
    
    print(f"\n--- Demonstrating {model_type} Model ---")
    env.render("Loaded Model - Initial State")
    
    for _ in range(50):
        if done: break
        action = np.argmax(model[state]) if is_rl_model else model.get(state, 0)
        state, _, done = env.step(action)
        env.render("Loaded Model - Path")
        time.sleep(0.2)
    print("Demonstration finished.")

def main():
    parser = argparse.ArgumentParser(description="Grid World Learning Experiment Runner", formatter_class=RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_run = subparsers.add_parser("run", help="Run a new training experiment.")
    parser_run.add_argument("--mode", choices=EXPERIMENTS, required=True, help="Select which experiment to run.")
    parser_run.add_argument("--render", action="store_true", help="Visualize final policy (only for single run).")
    parser_run.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment for robustness testing.")

    parser_demo = subparsers.add_parser("demonstrate", help="Demonstrate a pre-trained model.")
    parser_demo.add_argument("model_path", type=str, help="Path to the saved .pkl model file.")
    
    args = parser.parse_args()

    if args.command == "run":
        script_to_run = f"{args.mode}.py"
        run_script(script_to_run, render=args.render, runs=args.runs)
    elif args.command == "demonstrate":
        demonstrate_model(args.model_path)

if __name__ == "__main__":
    main()