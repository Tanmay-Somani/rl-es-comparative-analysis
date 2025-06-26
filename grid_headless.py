# grid_environment.py
import numpy as np
import time
import os

class GridEnvironment:
    """
    A simple 6x6 Grid World environment.
    'S' = Start, 'G' = Goal, 'X' = Obstacle, 'A' = Agent, '.' = Empty
    """
    def __init__(self):
        self.grid_size = 6
        self.start_pos = (0, 0)
        self.goal_pos = (5, 5)
        self.obstacle_pos = {(2, 2), (1, 4), (3, 1), (4, 4)}
        self.agent_pos = self.start_pos

        # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_space_size = len(self.actions)
        self.state_space_size = self.grid_size * self.grid_size

    def reset(self):
        """Resets the agent to the starting position."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action_index):
        """
        Performs an action and returns the new state, reward, and done flag.
        """
        # Calculate potential new position
        move = self.actions[action_index]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # Check for boundaries
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return self.agent_pos, -5, False # Penalize for hitting a wall

        # Check for obstacles
        if new_pos in self.obstacle_pos:
            return self.agent_pos, -50, True # Heavy penalty for hitting obstacle, episode ends

        # If move is valid, update agent position
        self.agent_pos = new_pos

        # Check for goal
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 100, True # High reward for reaching the goal, episode ends

        # Reward for a normal step
        return self.agent_pos, -1, False

    def render(self, title="Grid World"):
        """Prints the current state of the grid."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- {title} ---")
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[self.start_pos] = 'S'
        grid[self.goal_pos] = 'G'
        for obs in self.obstacle_pos:
            grid[obs] = 'X'
        grid[self.agent_pos] = 'A'
        
        for row in grid:
            print(" ".join(row))
        print("-" * (self.grid_size * 2 + 3))