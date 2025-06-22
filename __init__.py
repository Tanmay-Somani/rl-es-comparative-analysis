from gymnasium.envs.registration import register

register(
    id='TreasureMaze-v0',
    entry_point='treasure_maze_env:TreasureMazeEnv',
)
