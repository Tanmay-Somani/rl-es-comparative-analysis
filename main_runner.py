# main_runner.py
import argparse
import subprocess
import os
from argparse import RawTextHelpFormatter
from summary_logger import summarize_and_save

def run_script(script_name, render=False):
    """Executes a given script and then runs the summarizer on its logs."""
    try:
        print(f"\n Running experiment: {script_name}...")
        
        # Build the command to run the script
        command = ["python", script_name]
        if render:
            command.append("--render")
            
        # Execute the command
        # We use a timeout to prevent scripts from running indefinitely
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        
        # Print the script's output in real-time if needed, or after completion
        print("--- Script Output ---")
        print(process.stdout)
        if process.stderr:
            print("--- Script Errors ---")
            print(process.stderr)
        print(f"Experiment {script_name} completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}.")
        print("--- Error Output ---")
        print(e.stdout)
        print(e.stderr)
        return # Stop if the script failed
    except subprocess.TimeoutExpired as e:
        print(f" Timeout Error: {script_name} took too long to run and was terminated.")
        print(e.stdout)
        return

    # --- After successful run, generate summary ---
    experiment_name = script_name.replace('.py', '')
    log_file_path = os.path.join("logs", experiment_name, "training_log.csv")
    summarize_and_save(log_file_path, experiment_name)


def main():
    # Use RawTextHelpFormatter to allow for newlines and better formatting in help text
    parser = argparse.ArgumentParser(
        description="""
    ===================================================================
    Grid World Learning Experiment Runner
    ===================================================================
    This script runs and evaluates different learning algorithms on a 
    6x6 grid world environment.
    
    It will:
    1. Execute the specified learning algorithm.
    2. Log the performance (reward/fitness) to a CSV file.
    3. Generate a summary table and a performance plot from the log.
    """,
        formatter_class=RawTextHelpFormatter,
        epilog="""
    -------------------------------------------------------------------
    Examples:
    -------------------------------------------------------------------
    # Run centralized Q-learning and show the final agent's path
    python main_runner.py --mode centralized_rl --render
    
    # Run federated evolution strategies (without final visualization)
    python main_runner.py --mode federated_es
    
    After running, check the 'results/' folder for performance plots
    and summary files.
    """
    )
    
    parser.add_argument(
        "--mode",
        choices=["centralized_rl", "federated_rl", "centralized_es", "federated_es"],
        required=True,
        help="""Select which experiment to run:
  - centralized_rl: A single agent learns using Q-Learning.
  - federated_rl:   Multiple agents learn locally and a central server
                    aggregates their Q-tables (Federated Averaging).
  - centralized_es: A single population of policies is evolved to find
                    an optimal path.
  - federated_es:   Multiple isolated populations evolve and periodically
                    share their best individuals (champions).
"""
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="""Visualize the final learned policy after training completes.
The agent's path will be printed to the console.
"""
    )

    args = parser.parse_args()

    # Map the mode argument to the correct script file name
    mode_to_script = {
        "centralized_rl": "centralized_rl.py",
        "federated_rl": "federated_rl.py",
        "centralized_es": "centralized_es.py",
        "federated_es": "federated_es.py",
    }

    script_to_run = mode_to_script[args.mode]
    run_script(script_to_run, render=args.render)

if __name__ == "__main__":
    main()