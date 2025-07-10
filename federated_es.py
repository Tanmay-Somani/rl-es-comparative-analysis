# federated_evolution_strategies.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import copy
import argparse
import os
import csv
import pickle

# --- Hyperparameters ---
NUM_WORKERS = 4
FEDERATED_ROUNDS = 300
LOCAL_GENERATIONS = 5
POPULATION_PER_WORKER = 50
MUTATION_RATE = 0.05
ELITISM_COUNT = 3
TOURNAMENT_SIZE = 5

# --- Argument Parsing for Dynamic Logging ---
parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str, help="Path to save the training log CSV.")
parser.add_argument("--render", action="store_true", help="Render final policy.")
args, unknown = parser.parse_known_args()

# --- Setup for Logging ---
if args.log_file:
    LOG_FILE = args.log_file
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
else:
    LOG_DIR = "logs/federated_es"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'global_best_fitness', 'elapsed_time'])

# --- Policy and Fitness Functions (same as centralized for fair comparison) ---
def create_random_policy(env):
    return {(r, c): random.randint(0, 3) for r in range(env.grid_size) for c in range(env.grid_size)}

def evaluate_policy(policy, env):
    state = env.reset()
    done = False
    steps = 0
    max_steps = env.grid_size * env.grid_size * 1.5
    while not done and steps < max_steps:
        action = policy.get(state, 0)
        state, reward, done = env.step(action)
        steps += 1
    
    if state == env.goal_pos:
        return 1000.0 + (max_steps - steps) * 2
    else:
        final_pos, goal_pos = state, env.goal_pos
        manhattan_distance = abs(final_pos[0] - goal_pos[0]) + abs(final_pos[1] - goal_pos[1])
        fitness = -manhattan_distance * 10
        if done: fitness -= 500
        return fitness

# --- Worker Simulation ---
class Worker:
    def __init__(self, worker_id, env_config):
        self.id = worker_id
        self.env = GridEnvironment()
        self.env.set_config(env_config)
        self.population = [create_random_policy(self.env) for _ in range(POPULATION_PER_WORKER)]
        self.best_individual = None
        self.best_fitness = -float('inf')

    def evolve_locally(self):
        for _ in range(LOCAL_GENERATIONS):
            fitness_scores = sorted([(evaluate_policy(p, self.env), p) for p in self.population], key=lambda x: x[0], reverse=True)
            elites = [p for _, p in fitness_scores[:ELITISM_COUNT]]
            next_population = list(elites)
            while len(next_population) < POPULATION_PER_WORKER:
                parent = max(random.sample(fitness_scores, TOURNAMENT_SIZE), key=lambda x: x[0])[1]
                child = parent.copy()
                for state in child:
                    if random.random() < MUTATION_RATE:
                        child[state] = random.randint(0, 3)
                next_population.append(child)
            self.population = next_population
        final_scores = sorted([(evaluate_policy(p, self.env), p) for p in self.population], key=lambda x: x[0], reverse=True)
        self.best_fitness, self.best_individual = final_scores[0]

    def incorporate_champions(self, champions):
        self.population.sort(key=lambda p: evaluate_policy(p, self.env))
        num_to_replace = min(len(champions), len(self.population) // 4)
        for i in range(num_to_replace):
            self.population[i] = copy.deepcopy(random.choice(champions))

# --- Federated Evolution Loop ---
master_env = GridEnvironment()
master_env_config = master_env.get_config()
workers = [Worker(i, master_env_config) for i in range(NUM_WORKERS)]
global_best_policy = None
global_best_fitness = -float('inf')
start_time = time.time()

for round_num in range(FEDERATED_ROUNDS):
    print(f"--- Round {round_num+1}/{FEDERATED_ROUNDS} ---")
    champions = []
    for worker in workers:
        worker.evolve_locally()
        champions.append(worker.best_individual)
        if worker.best_fitness > global_best_fitness:
            global_best_fitness = worker.best_fitness
            global_best_policy = copy.deepcopy(worker.best_individual)
            print(f"** New Global Best Fitness: {global_best_fitness:.2f} (from Worker {worker.id}) **")
    
    elapsed_time = time.time() - start_time
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, global_best_fitness, elapsed_time])

    for worker in workers:
        other_champions = [c for c in champions if c is not worker.best_individual]
        if other_champions:
            worker.incorporate_champions(other_champions)

print("\n--- Federated Evolution Complete ---")

if global_best_policy:
    MODEL_DIR = "trained_models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_data = {'model': global_best_policy, 'env_config': master_env_config}
    model_path = os.path.join(MODEL_DIR, "federated_es.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Best global policy saved to {model_path}")

if args.render and global_best_policy:
    print("\n--- Demonstration of the Best Policy Found Globally ---")
    env = GridEnvironment()
    env.set_config(master_env_config)
    state = env.reset()
    done = False
    env.render("Federated ES - Final Path")
    for _ in range(50):
        if done: break
        action = global_best_policy.get(state,0)
        state, _, done = env.step(action)
        env.render("Federated ES - Final Path")
        time.sleep(0.2)
    print("Demonstration finished.")