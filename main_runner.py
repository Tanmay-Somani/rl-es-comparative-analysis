# main_runner.py
import argparse
import subprocess
import os
import time
import pickle
import numpy as np
from argparse import RawTextHelpFormatter

# Local import for demonstration logic
from grid_headless import GridEnvironment

def run_script(script_name, render=False):
    """Executes a given training script and then calls the analysis plotter."""
    try:
        print(f"\n>>> Running experiment: {script_name}...")
        
        command = ["python", script_name]
        if render:
            command.append("--render")
            
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        
        print("--- Script Output ---")
        print(process.stdout)
        if process.stderr:
            print("--- Script Errors ---")
            print(process.stderr)
        print(f">>> Experiment {script_name} completed successfully.")

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"!!! An error occurred while running {script_name}.")
        print("--- Error Output ---")
        print(e.stdout if hasattr(e, 'stdout') else "No stdout captured.")
        print(e.stderr if hasattr(e, 'stderr') else "No stderr captured.")
        return # Stop if the script failed

    # --- After successful run, call the new analysis plotter ---
    print("\n>>> Calling analysis plotter for a single report...")
    experiment_name = script_name.replace('.py', '')
    log_file_path = os.path.join("logs", experiment_name, "training_log.csv")
    
    # Call analysis_plotter.py as a separate process
    plotter_command = [
        "python", 
        "analysis_plotter.py", 
        "single", 
        "--log_file", log_file_path, 
        "--name", experiment_name
    ]
    subprocess.run(plotter_command)


# ... (The rest of the main_runner.py file remains unchanged) ...
# The 'demonstrate_model' function and the 'main' function with subparsers are the same.

def demonstrate_model(model_path):
    """Loads a saved model and demonstrates its performance on its original grid."""
    if not os.path.exists(model_path):
        print(f"!!! Error: Model file not found at {model_path}")
        return

    print(f">>> Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    env_config = data['env_config']
    env = GridEnvironment()
    env.set_config(env_config)
    state = env.reset()
    done = False
    
    is_rl_model = isinstance(model, np.ndarray)
    model_type = "Reinforcement Learning (Q-table)" if is_rl_model else "Evolutionary Strategy (Policy)"
    
    print("\n--- Demonstration ---")
    print(f"Model Type: {model_type}")
    
    env.render("Loaded Model - Initial State")
    path_length = 0
    max_path = env.grid_size * 10
    
    while not done and path_length < max_path:
        if is_rl_model:
            action = np.argmax(model[state])
        else:
            action = model.get(state, 0)
        
        state, _, done = env.step(action)
        env.render("Loaded Model - Path")
        path_length += 1
        print(f"Step: {path_length}, Position: {state}, Action: {action}")
        time.sleep(0.3)

    print("\n>>> Demonstration finished.")
    if done and state == env.goal_pos:
        print("Outcome: Goal was successfully reached!")
    else:
        print("Outcome: Agent did not reach the goal.")


def main():
    parser = argparse.ArgumentParser(
        description="Grid World Learning Experiment Runner & Demonstrator",
        formatter_class=RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    parser_run = subparsers.add_parser("run", help="Run a new training experiment.", formatter_class=RawTextHelpFormatter)
    parser_run.add_argument(
        "--mode",
        choices=["centralized_rl", "federated_rl", "centralized_es", "federated_es"],
        required=True,
        help="Select which experiment to run."
    )
    parser_run.add_argument("--render", action="store_true", help="Visualize final policy after training.")

    parser_demo = subparsers.add_parser("demonstrate", help="Demonstrate a pre-trained model.", formatter_class=RawTextHelpFormatter)
    parser_demo.add_argument("model_path", type=str, help="Path to the saved model .pkl file.")
    
    args = parser.parse_args()

    if args.command == "run":
        mode_to_script = {
            "centralized_rl": "centralized_rl.py", "federated_rl": "federated_rl.py",
            "centralized_es": "centralized_es.py", "federated_es": "federated_es.py",
        }
        script_to_run = mode_to_script[args.mode]
        run_script(script_to_run, render=args.render)
    
    elif args.command == "demonstrate":
        demonstrate_model(args.model_path)


if __name__ == "__main__":
    main()