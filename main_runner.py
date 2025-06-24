import argparse
import subprocess
from summary_logger import summarize_and_save

def run_script(script_name, render=False):
    try:
        print(f"Running: {script_name} ...")
        command = ["python", script_name]
        if render:
            command.append("--render")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while running {script_name}: {e}")
        return

    log_file_path = f"logs/{script_name.replace('.py', '')}/episode_rewards.csv"
    summarize_and_save(log_file_path, script_name.replace('.py', ''))

def main():
    parser = argparse.ArgumentParser(description="Run RL or ES experiments on Treasure Maze.")
    parser.add_argument(
        "--mode",
        choices=["centralized_rl", "federated_rl", "centralized_es", "federated_es"],
        required=True,
        help="Select which experiment to run from the federated and centralized es and rl models."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Visualize final policy after training (if supported)"
    )
    # parser.add_argument(
    #     "--help",
    #     action="help",
    #     help="Show this help message and exit",
    # )
    args = parser.parse_args()

    mode_to_script = {
        "centralized_rl": "centralized_rl.py",
        "federated_rl": "federated_rl.py",
        "centralized_es": "centralized_es.py",
        "federated_es": "federated_es.py",
    }

    run_script(mode_to_script[args.mode], render=args.render)

if __name__ == "__main__":
    main()
