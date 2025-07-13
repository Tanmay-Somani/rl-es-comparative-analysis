import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import os

# Register the Atari environments
try:
    gym.register_envs(ale_py)
except AttributeError:
    print("gym.register_envs is deprecated. Environments are likely already registered.")

# --- Configuration ---
# --- Federated Learning Configuration ---
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPISODES_PER_ROUND = 2 

# --- RL Configuration ---
ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
FRAME_STACK_SIZE = 4
REPLAY_BUFFER_SIZE = 50000  # Smaller buffer per client
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 200000 # Slower decay as training is distributed
TARGET_UPDATE_FREQ = 1000 # Target network updates are local to the client

# --- Logging and Saving ---
LOG_DIR = 'frl_breakout_logs'
LOG_FILE = os.path.join(LOG_DIR, 'frl_breakout_log.csv')
MODEL_SAVE_PATH = os.path.join(LOG_DIR, 'frl_global_model.weights.h5')
PLOT_SAVE_PATH = os.path.join(LOG_DIR, 'frl_training_plots.png')

# --- DQNAgent Class (The "Brain") ---
# This class is identical to the one in the original script
class DQNAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_network = self._build_dqn_model()
        self.target_q_network = self._build_dqn_model()
        self.update_target_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        self.loss_function = tf.keras.losses.Huber()

    def _build_dqn_model(self):
        """Builds the CNN model for the Q-network."""
        model = tf.keras.Sequential([
            layers.InputLayer(shape=(*INPUT_SHAPE, FRAME_STACK_SIZE)),
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.num_actions, activation='linear')
        ])
        return model

    def select_action(self, state, epsilon):
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        else:
            state_transposed = np.transpose(state, (1, 2, 0))
            q_values = self.q_network.predict(np.expand_dims(state_transposed, axis=0), verbose=0)
            return np.argmax(q_values[0])

    def train(self, replay_buffer, frame_count):
        """Trains the Q-network."""
        if len(replay_buffer) < BATCH_SIZE:
            return

        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        states = np.transpose(states, (0, 2, 3, 1)).astype(np.float32) / 255.0
        next_states = np.transpose(next_states, (0, 2, 3, 1)).astype(np.float32) / 255.0

        future_q_values = self.target_q_network.predict(next_states, verbose=0)
        target_q_values = rewards + (1 - dones.astype(np.float32)) * GAMMA * np.max(future_q_values, axis=1)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            action_masks = tf.one_hot(actions.astype(np.int32), self.num_actions)
            current_q_values = tf.reduce_sum(tf.multiply(q_values, action_masks), axis=1)
            loss = self.loss_function(target_q_values, current_q_values)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        if frame_count % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()

    def update_target_network(self):
        """Copies weights from Q-network to target Q-network."""
        self.target_q_network.set_weights(self.q_network.get_weights())
        
    def get_weights(self):
        return self.q_network.get_weights()
    
    def set_weights(self, weights):
        self.q_network.set_weights(weights)
        self.update_target_network()

# --- Client Class (The Player) ---
class Client:
    def __init__(self, client_id, num_actions):
        self.client_id = client_id
        self.env = gym.make(ENV_NAME, frameskip=1)
        self.env = gym.wrappers.AtariPreprocessing(self.env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
        self.env = gym.wrappers.FrameStackObservation(self.env, FRAME_STACK_SIZE)
        self.agent = DQNAgent(num_actions)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        
    def set_weights(self, weights):
        self.agent.set_weights(weights)

    def get_weights(self):
        return self.agent.get_weights()

    def local_train(self, num_episodes, global_frame_count):
        """Train the client's model locally."""
        print(f"--- Client {self.client_id}: Starting local training... ---")
        total_reward = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (global_frame_count / EPSILON_DECAY_STEPS))
                state_np = np.array(state)
                action = self.agent.select_action(state_np, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                self.replay_buffer.append((state_np, action, reward, np.array(next_state), terminated or truncated))
                state = next_state
                global_frame_count += 1
                
                self.agent.train(self.replay_buffer, global_frame_count)
            
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        print(f"--- Client {self.client_id}: Finished training. Avg Reward: {avg_reward:.2f} ---")
        return self.get_weights(), global_frame_count, avg_reward

# --- Central Server Class (The Coach) ---
class CentralServer:
    def __init__(self, num_actions):
        self.global_agent = DQNAgent(num_actions)

    def get_global_weights(self):
        return self.global_agent.get_weights()

    def aggregate_weights(self, client_weights):
        """Averages the weights from all clients (Federated Averaging)."""
        if not client_weights:
            return

        # Create a new list to hold the averaged weights
        averaged_weights = []
        
        # Get the number of layers in the model
        num_layers = len(client_weights[0])
        
        # For each layer...
        for i in range(num_layers):
            # Get the weights for this layer from all clients
            layer_weights = np.array([client_w[i] for client_w in client_weights])
            
            # Average the weights for this layer
            avg_layer_weight = np.mean(layer_weights, axis=0)
            averaged_weights.append(avg_layer_weight)
            
        # Update the server's global model
        self.global_agent.set_weights(averaged_weights)

    def save_model(self, path):
        self.global_agent.q_network.save_weights(path)
    
    def load_model(self, path):
        self.global_agent.q_network.load_weights(path)

# --- Utility Functions ---
def evaluate_global_model(server, num_episodes=5):
    """Evaluates the performance of the global model."""
    env = gym.make(ENV_NAME, frameskip=1, render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK_SIZE)
    
    agent = DQNAgent(env.action_space.n)
    agent.set_weights(server.get_global_weights())
    
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            state_np = np.array(state)
            action = agent.select_action(state_np, 0.05) # Use a small epsilon for evaluation
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
        
    env.close()
    return total_reward / num_episodes

def save_plots(log_data, save_path):
    df = pd.DataFrame(log_data)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(df['round'], df['average_evaluation_reward'], label='Avg Evaluation Reward (Global Model)')
    ax.set_xlabel('Federated Round')
    ax.set_ylabel('Reward')
    ax.set_title('Global Model Performance Over Federated Rounds')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Main Federated Training Loop ---
def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Initialize Server and Clients
    env_for_actions = gym.make(ENV_NAME)
    num_actions = env_for_actions.action_space.n
    env_for_actions.close()

    server = CentralServer(num_actions)
    clients = [Client(i, num_actions) for i in range(NUM_CLIENTS)]

    # --- Initialize or Resume Training State ---
    start_round = 0
    global_frame_count = 0
    log_data = []

    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(LOG_FILE):
        print("--- Resuming training from saved global model and log ---")
        server.load_model(MODEL_SAVE_PATH)
        df = pd.read_csv(LOG_FILE)
        log_data = df.to_dict('records')
        start_round = df['round'].iloc[-1]
        global_frame_count = df['global_frame_count'].iloc[-1]
        print(f"Resumed from Round: {start_round + 1}, Global Frames: {global_frame_count}")

    start_time = time.time()
    
    for round_num in range(start_round, NUM_ROUNDS):
        print(f"\n--- Starting Federated Round {round_num + 1}/{NUM_ROUNDS} ---")
        round_start_time = time.time()
        
        # 1. Distribute global model to all clients
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_weights(global_weights)

        # 2. Perform local training on all clients
        client_updates = []
        round_rewards = []
        for client in clients:
            updated_weights, global_frame_count, client_avg_reward = client.local_train(LOCAL_EPISODES_PER_ROUND, global_frame_count)
            client_updates.append(updated_weights)
            round_rewards.append(client_avg_reward)

        # 3. Aggregate weights on the server
        server.aggregate_weights(client_updates)
        print("--- Server: Aggregated client weights to update global model. ---")
        
        # 4. Evaluate the new global model
        avg_eval_reward = evaluate_global_model(server)
        
        round_duration = time.time() - round_start_time
        
        log_entry = {
            'round': round_num + 1,
            'global_frame_count': global_frame_count,
            'average_client_reward': np.mean(round_rewards),
            'average_evaluation_reward': avg_eval_reward,
            'time_elapsed_seconds': time.time() - start_time,
            'round_duration_seconds': round_duration
        }
        log_data.append(log_entry)
        print(f"--- Round {round_num + 1} Summary ---")
        print(f"Avg Client Reward: {np.mean(round_rewards):.2f}, Global Model Eval Reward: {avg_eval_reward:.2f}, Duration: {round_duration:.2f}s")

        # 5. Save logs, model, and plots periodically
        if (round_num + 1) % 5 == 0:
            try:
                pd.DataFrame(log_data).to_csv(LOG_FILE, index=False)
                server.save_model(MODEL_SAVE_PATH)
                save_plots(log_data, PLOT_SAVE_PATH)
                print(f"Log, global model, and plots saved at round {round_num + 1}")
            except Exception as e:
                print(f"Error saving log/model/plots: {e}")

    print("\n--- Federated Training Finished ---")
    total_training_time = time.time() - start_time
    print(f"Total Training Time: {total_training_time / 3600:.2f} hours")
    print(f"Log file saved to: {LOG_FILE}")
    print(f"Final Global Model saved to: {MODEL_SAVE_PATH}")
    print(f"Final Plots saved to: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()