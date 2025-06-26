# federated_evolution_strategies.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import copy

### --- NEW: Imports for logging and argument parsing --- ###
import argparse
import os
import csv

# --- Hyperparameters ---
NUM_WORKERS = 4
FEDERATED_ROUNDS = 40
LOCAL_GENERATIONS = 5
POPULATION_PER_WORKER = 50
MUTATION_RATE = 0.02
ELITISM_COUNT = 3

### --- NEW: Setup for Logging --- ###
LOG_DIR = "logs/federated_es"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

# Setup CSV logging
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'global_best_fitness'])

# --- Policy and Fitness Functions (same as centralized) ---
def create_random_policy(env):
    return {(r, c): random.randint(0, 3) for r in range(env.grid_size) for c in range(env.grid_size)}

def evaluate_policy(policy, env):
    state = env.reset()
    done = False
    steps = 0
    max_steps = env.grid_size * env.grid_size
    while not done and steps < max_steps:
        action = policy.get(state, 0)
        state, reward, done = env.step(action)
        steps += 1
    final_pos, goal_pos = state, env.goal_pos
    max_dist = (env.grid_size - 1) * 2
    manhattan_distance = abs(final_pos[0] - goal_pos[0]) + abs(final_pos[1] - goal_pos[1])
    fitness = (max_dist - manhattan_distance) * 10
    if final_pos == goal_pos: fitness += 1000
    fitness -= steps * 0.5
    if done and final_pos != goal_pos: fitness -= 200
    return fitness

# --- Worker Simulation ---
class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.env = GridEnvironment()
        self.population = [create_random_policy(self.env) for _ in range(POPULATION_PER_WORKER)]
        self.best_individual = None
        self.best_fitness = -float('inf')

    def evolve_locally(self):
        for _ in range(LOCAL_GENERATIONS):
            fitness_scores = sorted([(evaluate_policy(p, self.env), p) for p in self.population], key=lambda x: x[0], reverse=True)
            elites = [p for _, p in fitness_scores[:ELITISM_COUNT]]
            next_population = list(elites)
            while len(next_population) < POPULATION_PER_WORKER:
                parent = random.choice(elites)
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
workers = [Worker(i) for i in range(NUM_WORKERS)]
global_best_policy = None
global_best_fitness = -float('inf')

for round_num in range(FEDERATED_ROUNDS):
    print(f"--- Round {round_num+1}/{FEDERATED_ROUNDS} ---")
    champions = []
    round_best_fitness = -float('inf')

    for worker in workers:
        print(f"Worker {worker.id} evolving locally...")
        worker.evolve_locally()
        champions.append(worker.best_individual)
        
        if worker.best_fitness > global_best_fitness:
            global_best_fitness = worker.best_fitness
            global_best_policy = copy.deepcopy(worker.best_individual)
            print(f"** New Global Best Fitness: {global_best_fitness:.2f} (from Worker {worker.id}) **")
    
    ### --- NEW: Log the best fitness found in this round --- ###
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, global_best_fitness])

    print("Sharing champions among workers...")
    for worker in workers:
        other_champions = [c for c in champions if c is not worker.best_individual]
        if other_champions:
            worker.incorporate_champions(other_champions)

print("\n--- Federated Evolution Complete ---")

### --- MODIFIED: Final Demonstration is now conditional --- ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    args, unknown = parser.parse_known_args()

    if args.render and global_best_policy:
        print("\n--- Demonstration of the Best Policy Found Globally ---")
        env = GridEnvironment()
        state = env.reset()
        done = False
        env.render("Federated ES - Final Path")
        path_length = 0
        while not done and path_length < 25:
            action = global_best_policy[state]
            state, _, done = env.step(action)
            env.render("Federated ES - Final Path")
            path_length += 1
            print(f"Position: {state}")
            time.sleep(0.2)
        print("Demonstration finished.")
    elif args.render:
        print("Rendering skipped because no valid global policy was found.")