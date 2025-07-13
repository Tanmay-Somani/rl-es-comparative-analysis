# grid_headless.py
import numpy as np
import time
import os
import random

class GridEnvironment:
    """
    A simple 6x6 Grid World environment with randomized elements.
    'S' = Start, 'G' = Goal, 'X' = Obstacle, 'A' = Agent, '.' = Empty
    """
    ### MODIFIED: Initialization now randomizes the grid layout ###
    def __init__(self, grid_size=6, num_obstacles=4):
        self.grid_size = grid_size
        
        # Generate all possible coordinates
        all_coords = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        random.shuffle(all_coords)
        
        # Assign unique positions for goal, start, and obstacles
        self.goal_pos = all_coords.pop()
        self.start_pos = all_coords.pop()
        
        self.obstacle_pos = set()
        for _ in range(num_obstacles):
            if not all_coords: break # Stop if no more coordinates are available
            self.obstacle_pos.add(all_coords.pop())
            
        self.agent_pos = self.start_pos

        # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_space_size = len(self.actions)
        self.state_space_size = self.grid_size * self.grid_size

    ### NEW: Method to get the environment's configuration ###
    def get_config(self):
        """Returns the configuration of the grid."""
        return {
            'grid_size': self.grid_size,
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'obstacle_pos': self.obstacle_pos
        }

    ### NEW: Method to set the environment's configuration from a saved state ###
    def set_config(self, config):
        """Sets the grid configuration from a dictionary."""
        self.grid_size = config['grid_size']
        self.start_pos = config['start_pos']
        self.goal_pos = config['goal_pos']
        self.obstacle_pos = config['obstacle_pos']
        self.reset()


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
        # Note: Start pos might be the same as goal if grid is tiny, but our randomization prevents that.
        grid[self.start_pos] = 'S'
        grid[self.goal_pos] = 'G'
        for obs in self.obstacle_pos:
            grid[obs] = 'X'
        grid[self.agent_pos] = 'A'
        
        for row in grid:
            print(" ".join(row))
        print("-" * (self.grid_size * 2 + 3))