# RL-ES Comparative Analysis

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Gym-orange)![License](https.img.shields.io/badge/license-MIT-green)

A unified benchmark comparing Reinforcement Learning (RL) and Evolution Strategies (ES) algorithms under both centralized and federated learning paradigms. This project introduces a custom `Treasure Maze` environment, implements various training schemes, and provides a thorough analysis of their performance.

## 🧠 Project Overview

This project aims to evaluate and compare the performance of Reinforcement Learning (RL) and Evolution Strategies (ES) in both centralized and federated settings. By introducing a standardized environment and training protocols, we provide insights into how these powerful optimization methods scale, adapt, and perform in decentralized contexts, with a special focus on privacy implications and communication overhead.

## 📦 Key Features

- **Custom Environment:** A 5x5 `Treasure Maze` gridworld built with Gym compatibility, featuring sparse rewards and punishments to test exploration capabilities.
- **Diverse Agents:** Implementations for both RL and ES algorithms in two distinct training paradigms:
    - **Centralized:** PPO, DQN, and a standard ES algorithm.
    - **Federated:** Federated PPO and Federated ES, simulating decentralized training across multiple clients.
- **Flexible Architecture:** A standardized neural network architecture (2 hidden layers, 64 units each, ReLU activations) for fair comparison across all agents.
- **In-Depth Evaluation:** A comprehensive suite of metrics to analyze performance from multiple angles.

## 📈 Evaluation Metrics

Performance is assessed based on:

- **Convergence Speed:** Episodes required to reach the maximum reward target.
- **Exploration Efficiency:** Average episodic return over the course of training.
- **Training Stability:** Standard deviation of rewards across multiple independent runs.
- **Communication Overhead:** Number of communication rounds required in the federated setup.
- **Privacy Considerations:** A qualitative analysis of model weight sharing (ES) versus gradient sharing (RL).

## 🚀 Getting Started

### Prerequisites

To run this project, you will need:
- Python 3.10+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tanmay-Somani/rl-es-comparative-analysis.git
    cd rl-es-comparative-analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiments

You can run the different training scripts from the root directory:

- **Train a centralized RL agent (e.g., PPO):**
  ```bash
  python centralized_rl.py
  ```

### Directory Structure
rl-es-comparative-analysis/
│
├── environments/
│   └── treasure_maze.py    # Custom Gym-compatible grid environment
│
├── centralized_rl.py       # RL agent with centralized training
├── centralized_es.py       # ES agent with centralized training
├── federated_rl.py         # RL agent with federated training
├── federated_es.py         # ES agent with federated training
│
├── utils/
│   ├── logger.py           # Logging utilities
│   └── federated_core.py   # Core logic for federated learning
│
├── results/
│   └── plots/              # Learning curves and evaluation graphs
│
├── requirements.txt        # Project dependencies
└── README.md```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new algorithms, feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## 🙏 Acknowledgments

- This work builds upon foundational concepts from the fields of Reinforcement Learning and Evolutionary Computation.
- A special thanks to the creators of PyTorch and OpenAI Gym for their invaluable tools.
