import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TreasureMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(5, 5), custom_layout=None, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.layout = custom_layout if custom_layout is not None else self._default_layout()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32)  # (x, y)
        self.render_mode = render_mode
        self.start_pos = [0, 0]
        self.agent_pos = self.start_pos.copy()

    def _default_layout(self):
        layout = np.zeros(self.grid_size, dtype=np.uint8)
        layout[1, 1] = 1  # red = punishment
        layout[2, 2] = 2  # blue = reward
        return layout

    def reset(self, **kwargs):
        self.agent_pos = self.start_pos.copy()
        return np.array(self.agent_pos, dtype=np.float32), {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            x = max(x - 1, 0)
        elif action == 1:  # down
            x = min(x + 1, self.grid_size[0] - 1)
        elif action == 2:  # left
            y = max(y - 1, 0)
        elif action == 3:  # right
            y = min(y + 1, self.grid_size[1] - 1)

        self.agent_pos = [x, y]
        cell_type = self.layout[x, y]

        reward = 0
        if cell_type == 1:
            reward = -1  # punishment (red)
        elif cell_type == 2:
            reward = +2  # reward (blue)

        done = False 
        obs = np.array(self.agent_pos, dtype=np.float32)
        return obs, reward, done, False, {}

    def render(self):
        grid = np.copy(self.layout)
        x, y = self.agent_pos
        grid[x, y] = 3  # Agent = 3 for display
        print("\n".join(" ".join(self._cell_repr(cell) for cell in row) for row in grid))
        print()

    def _cell_repr(self, cell):
        return {
            0: "⬜",  # neutral
            1: "🟥",  # punishment
            2: "🟦",  # reward
            3: "🤖"   # agent
        }.get(cell, "❓")
