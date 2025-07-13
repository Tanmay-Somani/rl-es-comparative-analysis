# centralised_evolution_strategies.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import argparse
import os
import csv
import pickle

# --- Hyperparameters ---
GENERATIONS = 300
POPULATION_SIZE = 200
MUTATION_RATE = 0.05 
### TWEAK: Increased elitism to preserve good solutions on the new fitness scale
ELITISM_COUNT = 10 
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
    LOG_DIR = "logs/centralized_es"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['generation', 'best_fitness', 'elapsed_time'])

# --- Policy and Fitness Functions ---
def create_random_policy(env):
    return {(r, c): random.randint(0, env.action_space_size - 1) for r in range(env.grid_size) for c in range(env.grid_size)}

### TWEAK: The most important change. Fitness scale now matches RL reward scale.
def evaluate_policy(policy, env):
    """
    Fitness function scaled to match the environment's reward system for direct
    comparability with Reinforcement Learning.
    """
    state = env.reset()
    done = False
    steps = 0
    max_steps = env.grid_size * env.grid_size 
    
    while not done and steps < max_steps:
        action = policy.get(state, random.randint(0, env.action_space_size - 1))
        state, reward, done = env.step(action)
        steps += 1

    # --- SUCCESS CASE ---
    if state == env.goal_pos:
        # Maximum score is 100 (goal reward), minus a small penalty for each step.
        # This is directly analogous to the total reward in the RL agent.
        fitness = 100 - (steps * 0.5)
        return fitness
        
    # --- FAILURE CASE ---
    else:
        # If the agent crashed (hit an obstacle), its final reward would be -50.
        # We assign this fitness directly.
        if done:
            return -50
        
        # If it just timed out, penalize based on how far it was from the goal.
        # This keeps the score in a reasonable negative range.
        final_pos, goal_pos = state, env.goal_pos
        manhattan_distance = abs(final_pos[0] - goal_pos[0]) + abs(final_pos[1] - goal_pos[1])
        return -manhattan_distance * 2

def tournament_selection(fitness_scores, k):
    tournament_contenders = random.sample(fitness_scores, k)
    winner = max(tournament_contenders, key=lambda x: x[0])
    return winner[1]

def crossover(parent1, parent2):
    child_policy = {}
    for state in parent1:
        if random.random() < 0.5:
            child_policy[state] = parent1[state]
        else:
            child_policy[state] = parent2[state]
    return child_policy

def mutate(policy, mutation_rate, env):
    mutated_policy = policy.copy()
    for state in mutated_policy:
        if random.random() < mutation_rate:
            mutated_policy[state] = random.randint(0, env.action_space_size - 1)
    return mutated_policy

# --- Initialization ---
env = GridEnvironment() 
population = [create_random_policy(env) for _ in range(POPULATION_SIZE)]
start_time = time.time()

# --- Evolution Loop ---
for gen in range(GENERATIONS):
    fitness_scores = [(evaluate_policy(p, env), p) for p in population]
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    best_fitness = fitness_scores[0][0]
    avg_fitness = sum(f for f, p in fitness_scores) / len(fitness_scores)
    print(f"Gen {gen+1}/{GENERATIONS} | Best Fitness: {best_fitness:.2f} | Avg Fitness: {avg_fitness:.2f}")

    elapsed_time = time.time() - start_time
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([gen, best_fitness, elapsed_time])
    
    ### TWEAK: Early stopping condition now matches the new scale
    if best_fitness > 95:
        print("Excellent solution found, stopping early.")
        break

    next_population = []
    elites = [p for _, p in fitness_scores[:ELITISM_COUNT]]
    next_population.extend(elites)
    
    while len(next_population) < POPULATION_SIZE:
        parent1 = tournament_selection(fitness_scores, k=TOURNAMENT_SIZE)
        parent2 = tournament_selection(fitness_scores, k=TOURNAMENT_SIZE)
        child = crossover(parent1, parent2)
        child = mutate(child, MUTATION_RATE, env)
        next_population.append(child)
    population = next_population

# --- Final Evaluation and Saving ---
final_fitness_scores = sorted([(evaluate_policy(p, env), p) for p in population], key=lambda x: x[0], reverse=True)
best_policy = final_fitness_scores[0][1]
print(f"\n--- Centralized Evolution Complete ---\nBest policy fitness: {final_fitness_scores[0][0]:.2f}")

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_data = {'model': best_policy, 'env_config': env.get_config()}
model_path = os.path.join(MODEL_DIR, "centralized_es.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Best policy saved to {model_path}")

if args.render and best_policy:
    print("\n--- Demonstration of the Best Policy ---")
    state = env.reset()
    done = False
    env.render("Centralized ES - Final Path")
    for _ in range(50):
        action = best_policy.get(state)
        if done or action is None: break
        state, _, done = env.step(action)
        env.render("Centralized ES - Final Path")
        time.sleep(0.2)
    print("Demonstration finished.")