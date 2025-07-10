# centralised_evolution_strategies.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import argparse
import os
import csv
import pickle
import time
# --- Hyperparameters ---
### TWEAK: More generations for a harder problem
GENERATIONS = 100
POPULATION_SIZE = 100
### TWEAK: Crossover is the main driver of change, so mutation can be a fine-tuning mechanism.
MUTATION_RATE = 0.05 
### TWEAK: Keep only the absolute best, force others to be generated via crossover.
ELITISM_COUNT = 2 
TOURNAMENT_SIZE = 5
start_time=time.time()
# --- Setup for Logging ---
LOG_DIR = "logs/centralized_es"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['generation', 'best_fitness','elapsed_time'])

# --- Policy and Fitness Functions ---
def create_random_policy(env):
    return {(r, c): random.randint(0, env.action_space_size - 1) for r in range(env.grid_size) for c in range(env.grid_size)}

### TWEAK: Complete overhaul of the fitness function to be much stricter. ###
def evaluate_policy(policy, env):
    """
    A much stricter fitness function. High scores are only achievable by
    reaching the goal. Failing to reach the goal results in a negative score.
    """
    state = env.reset()
    done = False
    steps = 0
    max_steps = env.grid_size * env.grid_size * 1.5 # Allow some meandering

    while not done and steps < max_steps:
        action = policy.get(state, random.randint(0, env.action_space_size - 1))
        state, reward, done = env.step(action)
        steps += 1
    
    # --- Check for Success ---
    if state == env.goal_pos:
        # HUGE reward for reaching the goal.
        # The reward is higher for finishing in fewer steps.
        fitness = 1000.0 + (max_steps - steps) * 2
        return fitness
    
    # --- If Not Successful, Calculate Penalty ---
    else:
        # The primary penalty is for failing to reach the goal.
        # A smaller penalty is added based on the final distance.
        final_pos, goal_pos = state, env.goal_pos
        manhattan_distance = abs(final_pos[0] - goal_pos[0]) + abs(final_pos[1] - goal_pos[1])
        # The score will be negative, with closer failures being 'less bad'.
        fitness = -manhattan_distance * 10
        
        # Add a massive penalty if the agent failed by hitting an obstacle.
        if done: # 'done' is true but goal was not reached
            fitness -= 500
            
        return fitness

def tournament_selection(fitness_scores, k):
    tournament_contenders = random.sample(fitness_scores, k)
    winner = max(tournament_contenders, key=lambda x: x[0])
    return winner[1]

### TWEAK: Introduce Crossover for vastly improved exploration. ###
def crossover(parent1, parent2):
    """
    Creates a new child policy by combining genes (actions for states)
    from two parent policies.
    """
    child_policy = {}
    for state in parent1:
        # For each state, randomly choose which parent's action to inherit.
        if random.random() < 0.5:
            child_policy[state] = parent1[state]
        else:
            child_policy[state] = parent2[state]
    return child_policy

def mutate(policy, mutation_rate, env):
    """Applies random mutations to a policy."""
    mutated_policy = policy.copy()
    for state in mutated_policy:
        if random.random() < mutation_rate:
            mutated_policy[state] = random.randint(0, env.action_space_size - 1)
    return mutated_policy


# --- Initialization ---
env = GridEnvironment() 
population = [create_random_policy(env) for _ in range(POPULATION_SIZE)]

# --- Evolution Loop ---
for gen in range(GENERATIONS):
    fitness_scores = [(evaluate_policy(p, env), p) for p in population]
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    best_fitness = fitness_scores[0][0]
    
    # Calculate average fitness for monitoring population health
    avg_fitness = sum(f for f, p in fitness_scores) / len(fitness_scores)
    print(f"Gen {gen+1}/{GENERATIONS} | Best Fitness: {best_fitness:.2f} | Avg Fitness: {avg_fitness:.2f}")
    elapsed_time = time.time() - start_time

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([gen, best_fitness, elapsed_time])

    # Stop if a very good solution is found
    if best_fitness > 1100:
        print("Excellent solution found, stopping early.")
        break

    next_population = []
    # 1. Elitism: Keep the very best individuals
    elites = [p for _, p in fitness_scores[:ELITISM_COUNT]]
    next_population.extend(elites)
    
    # 2. Crossover & Mutation: Fill the rest of the population
    while len(next_population) < POPULATION_SIZE:
        # Select two distinct parents using tournaments
        parent1 = tournament_selection(fitness_scores, k=TOURNAMENT_SIZE)
        parent2 = tournament_selection(fitness_scores, k=TOURNAMENT_SIZE)
        
        # Create a child by combining their strategies
        child = crossover(parent1, parent2)
        
        # Apply a small chance of mutation to the child
        child = mutate(child, MUTATION_RATE, env)
        
        next_population.append(child)
        
    population = next_population

# --- Final Evaluation and Saving ---
final_fitness_scores = sorted([(evaluate_policy(p, env), p) for p in population], key=lambda x: x[0], reverse=True)
best_policy = final_fitness_scores[0][1]

print("\n--- Centralized Evolution Complete ---")
print(f"Best policy fitness: {final_fitness_scores[0][0]:.2f}")

# ... (Saving the model and demonstration parts remain the same) ...
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_data = {
    'model': best_policy,
    'env_config': env.get_config()
}
model_path = os.path.join(MODEL_DIR, "centralized_es.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Best policy saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    args, unknown = parser.parse_known_args()

    if args.render and best_policy:
        print("\n--- Demonstration of the Best Policy ---")
        state = env.reset()
        done = False
        env.render("Centralized ES - Final Path")
        path_length = 0
        while not done and path_length < 50:
            action = best_policy.get(state)
            if action is None: break
            state, _, done = env.step(action)
            env.render("Centralized ES - Final Path")
            path_length += 1
            print(f"Position: {state}")
            time.sleep(0.2)
        print("Demonstration finished.")