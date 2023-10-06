import gym
import numpy as np
from abc import ABC
from abc import abstractmethod

class IntrinsicRewardWrapper(gym.Wrapper, ABC):
    """
    Abstract class that adds intrisic reward wrapping to vectorized environment.
    """
    def __init__(self, env: gym.vector.AsyncVectorEnv):
        super().__init__(env)
        self.env = env
        self.num_envs = env.num_envs
        self.reset()

    def reset(self) -> list:
        """
        resets vectorized environments to initial state and returns initial observation from each.
        """
        self.dones = np.zeros(self.num_envs, dtype=bool)
        self.game_lens = np.zeros(self.num_envs, dtype=np.int16)
        self.extrinsic_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.intrinsic_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.states = self.env.reset()
        return self.states

    def step(self, actions: np.array) -> float:
        """
        Run one timestep in the vectorized environment.
        """
        self.actions = actions
        self.states, extrinsic_rewards, self.dones, self.infos = self.env.step(actions)
        self.game_lens += 1
        intrinsic_rewards = self.get_intrinsic_reward(self.states)
        self.intrinsic_rewards += intrinsic_rewards
        self.extrinsic_rewards += extrinsic_rewards
        return intrinsic_rewards + extrinsic_rewards

    @abstractmethod
    def get_intrinsic_reward(self, states: np.array) -> float:
        """
        Compute intrinsic reward from states.
        """

    def is_done(self) -> bool:
        """
        return true if any sub-environments are done.
        """
        return any(self.dones)

    def get_episode_data(self):
        """
        Generate episode data
        """
        return {
            "Extrinsic Reward": self.extrinsic_rewards.tolist(),
            "Intrinsic Reward": self.intrinsic_rewards.tolist(),
            "Trajectory Length": self.game_lens.tolist(),
            "Info": self.infos,
        }

    def get_render_data(self):
        """
        Generate render data
        """
        return {
            "Step": self.game_lens,
            "State": self.states,
            "Action": self.actions,
        }
