import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class Grid(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, grid_size=(6, 6), start_pos=(0, 0), goal_pos=(5, 5)):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.window_size = (grid_size[1] * self.cell_size, grid_size[0] * self.cell_size)
        
        self.layout = self._default_layout()
        self.start_pos = list(start_pos)
        self.goal_pos = list(goal_pos)

        self.agent_pos = self.start_pos.copy()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32)

        

        pygame.init()
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Grid Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)
        self.agent_img = pygame.image.load("robot_avatar.png").convert_alpha()
        self.agent_img = pygame.transform.scale(self.agent_img, (70, 70))
    def _animate_agent_move(self, start_pos, end_pos, steps=10):
        start_x, start_y = start_pos
        end_x, end_y = end_pos

        for i in range(1, steps + 1):
            t = i / steps
            interp_x = (1 - t) * start_x + t * end_x
            interp_y = (1 - t) * start_y + t * end_y

            self._draw_frame((interp_x, interp_y))
            pygame.time.delay(20)
            
    def _default_layout(self):
        layout = np.zeros(self.grid_size, dtype=np.uint8)

        # Mark traps (red)
        trap_positions = [(0, 1), (1, 3), (2, 1), (3, 3), (4, 2), (5, 4)]
        for r, c in trap_positions:
            layout[r, c] = 1

        # Optional bonus rewards (blue)
        reward_positions = [(1, 1), (2, 3), (3, 2), (4, 4)]
        for r, c in reward_positions:
            layout[r, c] = 2

        return layout

    def reset(self, **kwargs):
        self.agent_pos = self.start_pos.copy()
        return np.array(self.agent_pos, dtype=np.float32), {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0: x = max(x - 1, 0)       # up
        elif action == 1: x = min(x + 1, self.grid_size[0] - 1)  # down
        elif action == 2: y = max(y - 1, 0)       # left
        elif action == 3: y = min(y + 1, self.grid_size[1] - 1)  # right

        self.agent_pos = [x, y]
        cell_type = self.layout[x, y]

        reward, done = 0, False
        if self.agent_pos == self.goal_pos:
            reward, done = 10, True
        elif cell_type == 1:
            reward = -1
        elif cell_type == 2:
            reward = 2

        return np.array(self.agent_pos, dtype=np.float32), reward, done, False, {}

    def render(self):
        self.window.fill((255, 255, 255))  # white background

        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.layout[row, col]

                if [row, col] == self.start_pos or [row, col] == self.goal_pos:
                    color = (0, 0, 0)  # black (start & goal)
                elif cell_type == 1:
                    color = (255, 76, 76)  # red
                elif cell_type == 2:
                    color = (76, 174, 255)  # blue
                else:
                    color = (240, 240, 240)  # light gray

                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  # border

        # Draw agent
        x, y = self.agent_pos
        self.window.blit(self.agent_img, (y * self.cell_size + 20, x * self.cell_size + 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()
