<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# RL-ES-COMPARATIVE-ANALYSIS

<em></em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=default&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

`rl-es-comparative-analysis` is a comprehensive toolkit for training and comparing reinforcement learning (RL) agents using both centralized and federated evolutionary strategies (ES) and Q-learning, featuring a customizable grid-based environment and robust analysis tools.

**Why `rl-es-comparative-analysis`?**

This project provides a streamlined workflow for developing, training, and comparing RL algorithms. The core features include:

- **🔶 Centralized & Federated RL/ES:** Train agents using both centralized and distributed approaches, optimizing for performance and scalability.
- **🔷 Comparative Analysis:**  Easily compare performance metrics across different algorithms and configurations with automated report generation and visualization.
- **🔶 Customizable Grid Environment:** Experiment with a flexible grid-world environment, easily modifying parameters to suit your needs.
- **🔷  Intuitive Visualization:**  Understand results quickly with clear, informative plots of performance trends and moving averages.
- **🔶 Command-Line Interface:**  Execute experiments and analyze results efficiently via a user-friendly CLI.
- **🔷 Modular Design:**  Benefit from a well-structured codebase, promoting maintainability and extensibility.

---

## Features

<code>❯ REPLACE-ME</code>

---

## Project Structure

```sh
└── rl-es-comparative-analysis/
    ├── Atari_rl_es
    │   └── .gitignore
    ├── grid_es
    │   ├── .gitignore
    │   ├── __init__.py
    │   ├── analysis_plotter.py
    │   ├── centralized_es.py
    │   ├── centralized_rl.py
    │   ├── federated_es.py
    │   ├── federated_rl.py
    │   ├── grid.py
    │   ├── grid_headless.py
    │   ├── logs
    │   ├── main_runner.py
    │   ├── results
    │   ├── robot_avatar.png
    │   └── trained_models
    ├── Mountaincar_rl_es
    │   ├── .gitignore
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── communication_overhead_benchmark.png
    │   ├── convergence_speed_benchmark.png
    │   ├── cpu_usage_benchmark.png
    │   ├── es_distributed.py
    │   ├── fl_rl.py
    │   ├── memory_usage_benchmark.png
    │   ├── rl_non_fl.py
    │   ├── runner.py
    │   ├── training_time_benchmark.png
    │   └── utils.py
    ├── README.md
    └── requirements.txt
```

### Project Index

<details open>
	<summary><b><code>RL-ES-COMPARATIVE-ANALYSIS/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Requirements.txt specifies the projects dependencies<br>- It ensures the correct versions of crucial libraries, including Gymnasium for reinforcement learning environments, Stable Baselines3 for RL algorithms, PyTorch for deep learning, NumPy for numerical computation, and visualization tools like Matplotlib and Seaborn, are installed for successful execution<br>- Scikit-learn provides machine learning utilities, and tqdm offers progress bars.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- grid_es Submodule -->
	<details>
		<summary><b>grid_es</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ grid_es</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/analysis_plotter.py'>analysis_plotter.py</a></b></td>
					<td style='padding: 8px;'>- The analysis_plotter.py script processes experiment logs to generate performance reports<br>- It produces individual experiment summaries and plots, showing performance trends and moving averages<br>- Furthermore, it creates comparative analyses across multiple experiments, visualizing robustness, performance versus time, and privacy trade-offs<br>- These reports are saved to the <code>results</code> directory.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/centralized_es.py'>centralized_es.py</a></b></td>
					<td style='padding: 8px;'>- The <code>centralized_es.py</code> script implements a centralized evolutionary strategy algorithm to find an optimal policy for a grid-based environment<br>- It iteratively evolves a population of policies, evaluating their fitness based on a reward system, and selecting high-performing policies for reproduction and mutation<br>- The best policy is saved for later use or visualization<br>- The algorithm logs its progress and can optionally render the final policys performance.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/centralized_rl.py'>centralized_rl.py</a></b></td>
					<td style='padding: 8px;'>- The <code>centralized_rl.py</code> script trains a centralized reinforcement learning agent to navigate a grid environment<br>- It uses a Q-learning algorithm, logging training progress and saving the learned Q-table<br>- The agents performance is evaluated, and upon successful training, a demonstration of the optimal policy is optionally rendered<br>- The trained model is then saved for later use.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/federated_es.py'>federated_es.py</a></b></td>
					<td style='padding: 8px;'>- Federated evolution strategies trains a policy for a grid environment<br>- The code implements a distributed training algorithm, using multiple worker agents to evolve policies concurrently<br>- Each worker locally improves its policy via a genetic algorithm, then shares its best-performing policy with others<br>- This process iteratively refines a global best policy, which is saved for later use or demonstration<br>- Performance is logged to a CSV file.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/federated_rl.py'>federated_rl.py</a></b></td>
					<td style='padding: 8px;'>- Federated reinforcement learning trains a global Q-table for a grid environment<br>- Multiple worker agents independently learn local Q-tables, then these are averaged to update the global model<br>- The process iterates over numerous rounds, logging average global rewards<br>- Finally, the trained global Q-table and environment configuration are saved for later use or demonstration.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/grid.py'>grid.py</a></b></td>
					<td style='padding: 8px;'>- Grid.py<code> implements a grid-based environment for reinforcement learning, using Pygame for visualization<br>- It defines a </code>Grid<code> class that inherits from </code>gymnasium.Env`, creating a grid world with obstacles, rewards, and a goal<br>- The agent navigates this grid, receiving rewards or penalties based on its actions, aiming to reach the goal state<br>- The environment renders the grid visually, animating agent movements.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/grid_headless.py'>grid_headless.py</a></b></td>
					<td style='padding: 8px;'>- Grid_headless.py` defines a GridEnvironment class, a core component of a reinforcement learning project<br>- It simulates a randomized grid world, providing methods for agent movement, reward calculation, and environment configuration<br>- The class supports saving and loading environment states, enabling flexible experimentation and reproducibility within the broader project<br>- The environments dynamic nature enhances the complexity of the learning task.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/main_runner.py'>main_runner.py</a></b></td>
					<td style='padding: 8px;'>- Main_runner.py` orchestrates grid-world learning experiments<br>- It provides a command-line interface to execute various training scripts (centralized/federated RL and ES) and visualize results<br>- The runner manages multiple experimental runs for robustness analysis, and offers a model demonstration feature using saved policy or Q-tables<br>- Analysis plotting is automated post-training.</td>
				</tr>
			</table>
			<!-- logs Submodule -->
			<details>
				<summary><b>logs</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ grid_es.logs</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/logs/utils.py'>utils.py</a></b></td>
							<td style='padding: 8px;'>- The <code>utils.py</code> module provides utility functions for logging and visualizing training progress within the <code>grid_es</code> project<br>- It generates reward plots using Matplotlib, saving them to the logs directory, and appends episode and reward data to a specified log file<br>- These functions facilitate monitoring and analysis of reinforcement learning agent performance over training episodes.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- results Submodule -->
			<details>
				<summary><b>results</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ grid_es.results</b></code>
					<!-- single_reports Submodule -->
					<details>
						<summary><b>single_reports</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ grid_es.results.single_reports</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/results/single_reports/centralized_es_summary.txt'>centralized_es_summary.txt</a></b></td>
									<td style='padding: 8px;'>- Centralized_es_summary.txt reports key performance indicators from a single run of a genetic algorithm (likely within a larger grid-based evolutionary strategy)<br>- It summarizes the generation count, best fitness achieved, and the elapsed computation time, providing descriptive statistics (mean, standard deviation, percentiles) for analysis and comparison across different runs or configurations within the grid_es project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/results/single_reports/centralized_rl_summary.txt'>centralized_rl_summary.txt</a></b></td>
									<td style='padding: 8px;'>- Centralized RL performance results are summarized<br>- The report aggregates episode count, total reward, and elapsed time across 300 episodes<br>- Descriptive statistics, including mean, standard deviation, and percentiles, are provided to characterize the agents performance and runtime<br>- This data likely contributes to overall reinforcement learning algorithm evaluation within the larger grid-based simulation project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/results/single_reports/federated_es_summary.txt'>federated_es_summary.txt</a></b></td>
									<td style='padding: 8px;'>- Federated evolutionary strategy (ES) results are summarized<br>- The report presents statistical summaries of global best fitness, elapsed time, and round counts across multiple federated learning rounds<br>- Key statistics like mean, standard deviation, and percentiles are provided, offering a concise overview of the algorithms performance during the experiment<br>- This facilitates performance analysis within the broader federated ES framework.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/grid_es/results/single_reports/federated_rl_summary.txt'>federated_rl_summary.txt</a></b></td>
									<td style='padding: 8px;'>- Federated RL summary reports aggregate performance metrics from distributed reinforcement learning experiments<br>- The report summarizes average global reward, elapsed time per round, and their statistical distributions (mean, standard deviation, min, max, percentiles) across 300 training rounds<br>- This data provides insights into the algorithms convergence and efficiency within the broader federated learning framework.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- Mountaincar_rl_es Submodule -->
	<details>
		<summary><b>Mountaincar_rl_es</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ Mountaincar_rl_es</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/Mountaincar_rl_es/es_distributed.py'>es_distributed.py</a></b></td>
					<td style='padding: 8px;'>- Distributed Evolutionary Strategies (ES) are implemented for training an agent on the MountainCar-v0 environment<br>- The code employs a parallel processing approach, using multiple processes to evaluate different perturbed neural network weight sets concurrently<br>- It iteratively updates network parameters based on the performance of these perturbed networks, aiming to optimize agent performance over generations<br>- CPU and memory usage are monitored during training.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/Mountaincar_rl_es/fl_rl.py'>fl_rl.py</a></b></td>
					<td style='padding: 8px;'>- The <code>fl_rl.py</code> module simulates federated reinforcement learning for the MountainCar environment<br>- It implements a federated averaging algorithm, where multiple agents independently train Q-tables on local data, then aggregate their models to improve a global policy<br>- The module tracks performance metrics like CPU usage, memory consumption, training time, and average reward, facilitating analysis of the federated learning process.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/Mountaincar_rl_es/rl_non_fl.py'>rl_non_fl.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/Mountaincar_rl_es/runner.py'>runner.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='Research_Paper/rl-es-comparative-analysis/Mountaincar_rl_es/utils.py'>utils.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build rl-es-comparative-analysis from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../rl-es-comparative-analysis
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd rl-es-comparative-analysis
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![pip][pip-shield]][pip-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [pip-shield]: https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white -->
	<!-- [pip-link]: https://pypi.org/project/pip/ -->

	**Using [pip](https://pypi.org/project/pip/):**

	```sh
	❯ pip install -r requirements.txt
	```

### Usage

Run the project with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
python {entrypoint}
```

### Testing

Rl-es-comparative-analysis uses the {__test_framework__} test framework. Run the test suite with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
pytest
```

---

## Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://LOCAL/Research_Paper/rl-es-comparative-analysis/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/Research_Paper/rl-es-comparative-analysis/issues)**: Submit bugs found or log feature requests for the `rl-es-comparative-analysis` project.
- **💡 [Submit Pull Requests](https://LOCAL/Research_Paper/rl-es-comparative-analysis/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone C:\Users\Tanmay Somani\OneDrive\Desktop\Career\Research_Paper\rl-es-comparative-analysis
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/Research_Paper/rl-es-comparative-analysis/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Research_Paper/rl-es-comparative-analysis">
   </a>
</p>
</details>

---

## License

Rl-es-comparative-analysis is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---