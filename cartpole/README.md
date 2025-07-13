# CartPole RL/ES Comparative Analysis

This directory contains implementations of Reinforcement Learning (RL) and Evolutionary Strategies (ES) for the CartPole-v1 environment, including both centralized and federated learning approaches.

## Overview

The CartPole environment is a classic control problem where the goal is to balance a pole on a moving cart. The agent receives a reward of +1 for every timestep the pole remains upright, and the episode ends when the pole falls or the cart moves too far from the center.

## Implementations

### 1. Non-Federated Reinforcement Learning (`rl_non_fl.py`)
- **Algorithm**: Q-Learning with discretized state space
- **State Space**: 4-dimensional continuous state (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 discrete actions (push left, push right)
- **Features**:
  - Discretizes the continuous state space into bins
  - Uses epsilon-greedy exploration strategy
  - Updates Q-table using standard Q-learning update rule

### 2. Distributed Evolutionary Strategies (`es_distributed.py`)
- **Algorithm**: Distributed Evolutionary Strategies with neural network policies
- **Network Architecture**: 4 → 64 → 32 → 2 (tanh activation)
- **Features**:
  - Uses multiprocessing for parallel evaluation
  - Implements SGD optimizer with momentum
  - Uses utility-based ranking for parameter updates

### 3. Federated Reinforcement Learning (`fl_rl.py`)
- **Algorithm**: Federated Q-Learning with federated averaging
- **Features**:
  - Multiple clients train locally using Q-learning
  - Global model aggregation using federated averaging
  - Simulates communication overhead between clients and server

### 4. Federated Evolutionary Strategies (`fl_es.py`)
- **Algorithm**: Federated Evolutionary Strategies
- **Features**:
  - Multiple clients run local ES training
  - Global parameter aggregation using federated averaging
  - Combines benefits of ES with federated learning

## Environment Details

- **Environment**: CartPole-v1
- **State Space**: 4 continuous variables
  - Cart Position: [-4.8, 4.8]
  - Cart Velocity: [-∞, ∞] (clipped to [-10, 10] for discretization)
  - Pole Angle: [-0.418, 0.418] radians
  - Pole Angular Velocity: [-∞, ∞] (clipped to [-10, 10] for discretization)
- **Action Space**: 2 discrete actions (0: push left, 1: push right)
- **Reward**: +1 for each timestep the pole remains upright
- **Episode Length**: Maximum 500 timesteps

## Usage

### Running the Complete Analysis

```bash
cd rl-es-comparative-analysis/cartpole
python runner.py
```

This will run all four implementations and generate comparison plots.

### Running Individual Methods

```python
# Non-federated RL
from rl_non_fl import run_rl_non_fl
log_data = {'rl_non_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': []}}
log_data = run_rl_non_fl(log_data)

# Distributed ES
from es_distributed import run_es_distributed
log_data = {'es_distributed': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': []}}
log_data = run_es_distributed(log_data)

# Federated RL
from fl_rl import run_fl_rl
log_data = {'rl_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': [], 'communication_overhead': []}}
log_data = run_fl_rl(log_data)

# Federated ES
from fl_es import run_fl_es
log_data = {'es_fl': {'cpu_usage': [], 'memory_usage': [], 'training_time': [], 'convergence_speed': [], 'communication_overhead': []}}
log_data = run_fl_es(log_data)
```

## Output

The analysis generates several plots:

1. **CPU Usage Benchmark**: Comparison of CPU utilization across methods
2. **Memory Usage Benchmark**: Comparison of memory consumption
3. **Training Time Benchmark**: Comparison of training time progression
4. **Convergence Speed Benchmark**: Comparison of reward progression
5. **Communication Overhead Benchmark**: Communication costs for federated methods
6. **Comprehensive Comparison**: Combined view of all metrics

## Key Differences from MountainCar

1. **State Space**: CartPole has 4 continuous state variables vs MountainCar's 2
2. **Action Space**: CartPole has 2 discrete actions vs MountainCar's 3
3. **Reward Structure**: CartPole gives +1 per timestep vs MountainCar's sparse rewards
4. **Episode Length**: CartPole episodes can last up to 500 timesteps vs MountainCar's 200
5. **Discretization**: More complex state space discretization for Q-learning

## Performance Expectations

- **RL Non-FL**: Should achieve high rewards quickly due to simple reward structure
- **ES Distributed**: May take longer to converge but can find good policies
- **RL FL**: Should show communication overhead but maintain performance
- **ES FL**: Combines benefits of ES with federated learning advantages

## Dependencies

- gymnasium
- numpy
- matplotlib
- psutil
- tqdm
- multiprocessing (built-in)

## Notes

- The federated implementations simulate the federated learning process
- Communication overhead is measured in bytes transferred
- All methods use the same evaluation protocol for fair comparison
- Results may vary depending on hardware and system load 