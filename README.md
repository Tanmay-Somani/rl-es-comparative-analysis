# RL-ES Comparative Analysis

A unified benchmark comparing Reinforcement Learning (RL) and Evolution Strategies (ES) algorithms under centralized and federated paradigms. This project includes a custom `Treasure Maze` environment, federated learning implementations, and a thorough analysis of convergence behavior, exploration efficiency, and communication overhead.

## 🧠 Project Overview

This project aims to evaluate and compare the performance of RL and ES algorithms in both centralized and federated learning settings. By introducing a standardized environment and training protocols, we provide insights into how these methods scale and adapt in decentralized contexts.

## 📦 Components

- **Treasure Maze Environment**  
  A 5x5 gridworld built with Gym compatibility, including:
  - Red cells (punishment)
  - Blue cells (reward)
  - White cells (neutral)

- **Agents**
  - Centralized PPO
  - Centralized DQN
  - Centralized ES
  - Federated PPO
  - Federated ES

- **Neural Network Architecture**
  - 2 hidden layers
  - 64 units per layer
  - ReLU activations

## 📈 Evaluation Metrics

- **Convergence**  
  Episodes required to reach maximum reward target.

- **Exploration Efficiency**  
  Average episodic return over time.

- **Stability**  
  Standard deviation across multiple training runs.

- **Communication Overhead**  
  Number of communication rounds in federated setups.

- **Privacy Considerations**  
  Comparison of model weight sharing vs. gradient sharing.

## 📁 Directory Structure

# 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Gym
- NumPy
- PyTorch
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```