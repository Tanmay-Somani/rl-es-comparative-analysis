# Assault RL/ES Comparative Analysis

This directory contains implementations of Reinforcement Learning (RL) and Evolutionary Strategies (ES) for the Atari Assault game (ALE/Assault-v5), including both centralized and federated learning approaches.

## Overview

The Assault environment is an Atari 2600 game where the agent controls a vehicle to destroy enemies and dodge attacks. This project compares RL and ES methods, both in centralized and federated settings, using RAM observations for tractability.

## Implementations

### 1. Non-Federated RL (`rl_non_fl.py`)
- **Algorithm:** Q-Learning with discretized RAM state (first 4 bytes, 8 bins each)
- **Action Space:** Discrete(7)
- **Observation:** RAM (128 bytes)
- **Features:**
  - Epsilon-greedy exploration
  - Q-table update rule

### 2. Distributed Evolutionary Strategies (`es_distributed.py`)
- **Algorithm:** Distributed ES with neural network policy
- **Network:** 128 (RAM) → 64 → 32 → 7 (actions)
- **Features:**
  - Multiprocessing for parallel evaluation
  - SGD optimizer with momentum
  - Utility-based ranking for updates

### 3. Federated RL (`fl_rl.py`)
- **Algorithm:** Federated Q-Learning (same as non-FL, but with multiple clients)
- **Features:**
  - Multiple clients train locally
  - Global model aggregation (federated averaging)
  - Communication overhead tracking

### 4. Federated ES (`fl_es.py`)
- **Algorithm:** Federated ES (clients run local ES, then average parameters)
- **Features:**
  - Multiple clients, local ES training
  - Global parameter aggregation
  - Communication overhead tracking

## Usage

### Running the Complete Analysis

```bash
cd rl-es-comparative-analysis/assault
python runner.py
```

This will run all four implementations and generate comparison plots in the current directory.

### Output

The analysis generates several plots:

- **CPU Usage Benchmark**
- **Memory Usage Benchmark**
- **Training Time Benchmark**
- **Convergence Speed Benchmark**
- **Communication Overhead Benchmark** (for federated methods)
- **Comprehensive Comparison**

## Notes

- All code and output files are contained within the `assault` folder.
- RAM observations are used for tractability in Q-learning and ES.
- Communication overhead is measured in bytes transferred.
- All methods use the same evaluation protocol for fair comparison.
- Results may vary depending on hardware and system load.

## Dependencies

- gymnasium
- numpy
- matplotlib
- psutil
- tqdm
- multiprocessing (built-in)

Install dependencies with:

```bash
pip install gymnasium[atari] numpy matplotlib psutil tqdm
```

## Citation

If you use this code, please cite the ALE and Gymnasium projects. 