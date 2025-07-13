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
ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
FRAME_STACK_SIZE = 4
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 1000000
TARGET_UPDATE_FREQ = 10000
TRAINING_FRAMES = 500000
LOG_DIR = 'dqn_breakout_logs'
LOG_FILE = os.path.join(LOG_DIR, 'dqn_breakout_log.csv')
MODEL_SAVE_PATH = os.path.join(LOG_DIR, 'dqn_breakout_model.weights.h5')
PLOT_SAVE_PATH = os.path.join(LOG_DIR, 'training_plots.png')

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
            layers.InputLayer(shape=(*INPUT_SHAPE, FRAME_STACK_SIZE)), # (84, 84, 4)
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

    def train(self, replay_buffer):
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

    def update_target_network(self):
        """Copies weights from Q-network to target Q-network."""
        self.target_q_network.set_weights(self.q_network.get_weights())

    def load_model(self, path):
        """Loads weights from a file."""
        self.q_network.load_weights(path)
        self.update_target_network()

    def save_model(self, path):
        """Saves the Q-network weights."""
        self.q_network.save_weights(path)


def save_plots(log_data, save_path):
    df = pd.DataFrame(log_data)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Episode Reward
    axs[0].plot(df['episode'], df['episode_reward'], label='Episode Reward')
    axs[0].plot(df['episode'], df['average_reward_last_100'], label='Avg Reward (100 episodes)', linestyle='--')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Episode Rewards and Running Average')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Epsilon Decay
    axs[1].plot(df['frame_count'], df['epsilon'], label='Epsilon')
    axs[1].set_xlabel('Frame Count')
    axs[1].set_ylabel('Epsilon')
    axs[1].set_title('Epsilon Decay Over Time')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    env = gym.make(ENV_NAME, frameskip=1, render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK_SIZE)

    num_actions = env.action_space.n
    agent = DQNAgent(num_actions)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    frame_count = 0
    episode_count = 0
    log_data = []
    episode_reward_history = []
    
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(LOG_FILE):
        print("--- Resuming training from saved model and log ---")
        agent.load_model(MODEL_SAVE_PATH)
        
        df = pd.read_csv(LOG_FILE)
        log_data = df.to_dict('records')
        frame_count = df['frame_count'].iloc[-1]
        episode_count = df['episode'].iloc[-1]
        if 'average_reward_last_100' in df.columns:
            past_rewards = df['episode_reward'].tail(100).tolist()
            episode_reward_history.extend(past_rewards)

        print(f"Resumed from Episode: {episode_count}, Frame Count: {frame_count}")
    else:
        print("--- Starting new training session ---")

    epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (frame_count / EPSILON_DECAY_STEPS))

    start_time = time.time()

    while frame_count < TRAINING_FRAMES:
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        episode_start_time = time.time()
        episode_frame_count = 0

        while not terminated and not truncated:
            state_np = np.array(state)
            action = agent.select_action(state_np, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            replay_buffer.append((state_np, action, reward, np.array(next_state), terminated or truncated))
            state = next_state
            
            frame_count += 1
            episode_frame_count +=1
            epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (frame_count / EPSILON_DECAY_STEPS))

            if frame_count > BATCH_SIZE:
                agent.train(replay_buffer)

            if frame_count % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                print(f"--- Target network updated at frame {frame_count} ---")
        
        episode_count += 1
        episode_reward_history.append(episode_reward)
        avg_reward = np.mean(episode_reward_history[-100:])

        episode_duration = time.time() - episode_start_time
        speed_fps = episode_frame_count / episode_duration if episode_duration > 0 else 0

        log_entry = {
            'episode': episode_count,
            'frame_count': frame_count,
            'episode_reward': episode_reward,
            'average_reward_last_100': avg_reward,
            'epsilon': epsilon,
            'time_elapsed_seconds': time.time() - start_time,
            'speed_fps': speed_fps
        }
        log_data.append(log_entry)
        print(f"Episode: {episode_count}, Frames: {frame_count}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Speed: {speed_fps:.2f} FPS")

        if episode_count % 5 == 0:
            try:
                pd.DataFrame(log_data).to_csv(LOG_FILE, index=False)
                agent.save_model(MODEL_SAVE_PATH)
                save_plots(log_data, PLOT_SAVE_PATH)
                print(f"Log, model, and plots saved at episode {episode_count}")
            except Exception as e:
                print(f"Error saving log/model/plots: {e}")

    env.close()
    df = pd.DataFrame(log_data)
    df.to_csv(LOG_FILE, index=False)
    
    total_training_time = time.time() - start_time
    final_avg_reward = np.mean(episode_reward_history[-100:])
    best_avg_reward = df['average_reward_last_100'].max()
    
    print("\n--- Training Finished ---")
    print("--- Benchmark Summary ---")
    print(f"Total Training Time: {total_training_time / 3600:.2f} hours")
    print(f"Total Frames: {frame_count}")
    print(f"Total Episodes: {episode_count}")
    print(f"Final Average Reward (last 100 episodes): {final_avg_reward:.2f}")
    print(f"Best Average Reward (100 episodes): {best_avg_reward:.2f}")
    print(f"Log file saved to: {LOG_FILE}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Plots saved to: {PLOT_SAVE_PATH}")
    save_plots(log_data, PLOT_SAVE_PATH)

if __name__ == "__main__":
    main()