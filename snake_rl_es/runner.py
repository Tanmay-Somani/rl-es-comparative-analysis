# runner.py
import time
import pandas as pd
import psutil
from tqdm import tqdm
import numpy as np

from snake_game import SnakeGame
from rl_agent import DQNAgent
from es_agent import ESAgent
from fl_rl_agent import FLAgent
from utils import plot_results

def run_rl():
    log = []
    start_time = time.time()
    
    game = SnakeGame()
    state_size = len(game.get_state())
    action_size = 3 # [straight, right, left]
    agent = DQNAgent(state_size, action_size)
    
    num_episodes = 250
    
    for e in tqdm(range(num_episodes), desc="Training RL Agent"):
        game.reset()
        state = game.get_state()
        state = np.reshape(state, [1, state_size])
        
        for t in range(5000): # Max steps per episode
            action_idx = agent.get_action(state)
            
            final_move = [0, 0, 0]
            final_move[action_idx] = 1

            reward, done, score = game.play_step(final_move)
            next_state = game.get_state()
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            
            if done:
                agent.update_target_model()
                break
        
        agent.replay()
        
        # Logging
        log.append({
            'episode': e,
            'score': score,
            'time_elapsed': time.time() - start_time,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent
        })

    pd.DataFrame(log).to_csv('rl_benchmark.csv', index=False)
    plot_results('rl_benchmark.csv', 'Reinforcement Learning Performance')

def run_es():
    log = []
    start_time = time.time()

    game = SnakeGame()
    state_size = len(game.get_state())
    action_size = 3
    agent = ESAgent(state_size, action_size, population_size=20, sigma=0.1, learning_rate=0.03)

    num_generations = 5

    for g in tqdm(range(num_generations), desc="Training ES Agent"):
        population = agent.generate_population()
        rewards = []
        total_score = 0
        
        for model in population:
            game.reset()
            state = game.get_state()
            
            for _ in range(2000): # Max steps
                action_idx = model.get_action(state)
                final_move = [0, 0, 0]
                final_move[action_idx] = 1
                
                _, done, score = game.play_step(final_move)
                state = game.get_state()
                if done:
                    break
            rewards.append(score)
            total_score += score
        
        agent.evolve(rewards)
        
        log.append({
            'generation': g,
            'score': total_score / agent.population_size, # Average score
            'time_elapsed': time.time() - start_time,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'communication_cost': 0 # No communication in this simple ES
        })

    pd.DataFrame(log).to_csv('es_benchmark.csv', index=False)
    plot_results('es_benchmark.csv', 'Evolutionary Strategies Performance')

def run_fl_rl():
    log = []
    start_time = time.time()

    # FL Hyperparameters
    num_clients = 5
    num_rounds = 50
    local_epochs = 5
    
    # Initialize FL system
    game_env = SnakeGame() # A dummy env to get state size
    state_size = len(game_env.get_state())
    action_size = 3
    fl_system = FLAgent(state_size, action_size, num_clients)

    # Simulate client environments
    client_envs = [SnakeGame() for _ in range(num_clients)]

    for r in tqdm(range(num_rounds), desc="Federated Learning Rounds"):
        round_scores = []
        for client_idx, client in enumerate(fl_system.clients):
            client_env = client_envs[client_idx]
            client_score = 0
            
            # Local training
            for e in range(local_epochs):
                client_env.reset()
                state = client_env.get_state()
                state = np.reshape(state, [1, state_size])

                for _ in range(2000):
                    action_idx = client.agent.get_action(state)
                    final_move = [0, 0, 0]
                    final_move[action_idx] = 1

                    reward, done, score = client_env.play_step(final_move)
                    next_state = client_env.get_state()
                    next_state = np.reshape(next_state, [1, state_size])
                    
                    client.agent.remember(state, action_idx, reward, next_state, done)
                    state = next_state
                    
                    if done:
                        client_score = score
                        break
                
                client.agent.replay()
            
            round_scores.append(client_score)
        
        # Aggregate models and get communication cost
        comm_cost = fl_system.global_round()
        
        log.append({
            'round': r,
            'score': np.mean(round_scores),
            'time_elapsed': time.time() - start_time,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'communication_cost': comm_cost
        })

    pd.DataFrame(log).to_csv('fl_rl_benchmark.csv', index=False)
    plot_results('fl_rl_benchmark.csv', 'Federated Reinforcement Learning Performance')


if __name__ == '__main__':
    print("Choose the agent to run:")
    print("1: Reinforcement Learning (RL)")
    print("2: Evolutionary Strategies (ES)")
    print("3: Federated Reinforcement Learning (FL-RL)")
    
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        run_rl()
    elif choice == '2':
        run_es()
    elif choice == '3':
        run_fl_rl()
    else:
        print("Invalid choice. Please run the script again.")