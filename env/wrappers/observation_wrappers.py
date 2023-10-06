import gym
import numpy as np


class AtariObservationWrapper(gym.ObservationWrapper):
    """
    Transforms state space for Atari environment.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(40, 40),
            dtype=np.float32,
        )

    def observation(self, obs: np.array):
        obs = obs[35:195:4, ::4, 0] # frame skip
        return (obs / 255).astype(np.float32)


class GridWorldObservationWrapper(gym.ObservationWrapper):
    """
    Transforms state space for Grid world custom environments.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(env.grid_size, env.grid_size),
            dtype=np.float32,
        )

    def observation(self, obs: np.array):
        return (obs / 255).astype(np.float32)
