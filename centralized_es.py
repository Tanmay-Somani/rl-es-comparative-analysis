# centralised_evolution_strategies.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment

### --- NEW: Imports for logging and argument parsing --- ###
import argparse
import os
import csv

# --- Hyperparameters ---
GENERATIONS = 150
POPULATION_SIZE = 200
MUTATION_RATE = 0.02
ELITISM_COUNT = 10

### --- NEW: Setup for Logging --- ###
LOG_DIR = "logs/centralized_es"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

# Setup CSV logging
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['generation', 'best_fitness'])

# --- Policy and Fitness Functions ---
def create_random_policy(env):
    return {(r, c): random.randint(0, env.action_space_size - 1) for r in range(env.grid_size) for c in range(env.grid_size)}

def evaluate_policy(policy, env):
    state = env.reset()
    done = False
    steps = 0
    max_steps = env.grid_size * env.grid_size
    while not done and steps < max_steps:
        action = policy[state]
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

# --- Initialization ---
env = GridEnvironment()
population = [create_random_policy(env) for _ in range(POPULATION_SIZE)]

# --- Evolution Loop ---
for gen in range(GENERATIONS):
    fitness_scores = [(evaluate_policy(p, env), p) for p in population]
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    best_fitness = fitness_scores[0][0]
    
    print(f"Generation {gen+1}/{GENERATIONS} | Best Fitness: {best_fitness:.2f}")

    ### --- NEW: Log best fitness for the generation --- ###
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([gen, best_fitness])
    
    if best_fitness > 900:
        print("Good solution found, stopping early.")
        break

    next_population = []
    elites = [p for _, p in fitness_scores[:ELITISM_COUNT]]
    next_population.extend(elites)
    while len(next_population) < POPULATION_SIZE:
        parent = random.choice(elites)
        child = parent.copy()
        for state in child:
            if random.random() < MUTATION_RATE:
                child[state] = random.randint(0, env.action_space_size - 1)
        next_population.append(child)
    population = next_population

final_fitness_scores = sorted([(evaluate_policy(p, env), p) for p in population], key=lambda x: x[0], reverse=True)
best_policy = final_fitness_scores[0][1]

print("\n--- Centralized Evolution Complete ---")
print(f"Best policy fitness: {final_fitness_scores[0][0]:.2f}")

### --- MODIFIED: Final Demonstration is now conditional --- ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    args, unknown = parser.parse_known_args()

    if args.render:
        print("\n--- Demonstration of the Best Policy ---")
        state = env.reset()
        done = False
        env.render("Centralized ES - Final Path")
        path_length = 0
        while not done and path_length < 25:
            action = best_policy[state]
            state, _, done = env.step(action)
            env.render("Centralized ES - Final Path")
            path_length += 1
            print(f"Position: {state}")
            time.sleep(0.2)
        print("Demonstration finished.")