import argparse
import gym
import numpy as np
from scipy.stats import entropy
from env.wrappers import IntrinsicRewardWrapper

class MiceWrapper(IntrinsicRewardWrapper):
    """
    General wrapper that adds MICE to a vectorized environment.
    """
    def __init__(self, env: gym.vector.AsyncVectorEnv, args: argparse.Namespace):
        self.entropy_coef = args.entropy_coef
        self.mutual_coef = args.mutual_coef
        self._state_dim = np.prod(env.single_observation_space.shape)
        self._state_idx = np.arange(self._state_dim)
        super().__init__(env)

    def reset(self) -> list:
        """
        resets vectorized environments to initial state and returns initial observation from each.
        """
        self.current_entropy = np.zeros(self.num_envs, dtype=np.float32)
        self.current_mutual_info = np.zeros(self.num_envs, dtype=np.float32)
        self.last_entropy = np.zeros(self.num_envs, dtype=np.float32)
        self.last_mutual_info = np.zeros(self.num_envs, dtype=np.float32)
        self.count_matrix = None
        self.unique = []
        states = super().reset()
        self.mice(states)
        return states

    def get_intrinsic_reward(self, states: np.array) -> float:
        return self.mice(states)

    def mice(self, states: np.array) -> float:
        """
        Compute mice exploration bonus.
        """
        r_intrinsic = np.empty(self.num_envs)
        for idx, state in enumerate(states):
            self._update(state, idx)
            self.current_entropy[idx] = self._entropy(self.count_matrix[idx])
            self.current_mutual_info[idx] = self._mutual_info(idx)
            delta_entropy = self.current_entropy[idx] - self.last_entropy[idx]
            delta_mutual_info = self.current_mutual_info[idx] - self.last_mutual_info[idx]
            r_intrinsic[idx] = self.entropy_coef * delta_entropy - self.mutual_coef * delta_mutual_info
            self.last_entropy[idx] = self.current_entropy[idx]
            self.last_mutual_info[idx] = self.current_mutual_info[idx]
        return r_intrinsic

    def _update(self, state: np.array, idx: int) -> None:
        """
        Update internal state.
        """
        state = (state * 256).astype(np.uint8).ravel()
        state = np.concatenate((self.unique, state))
        unique, unique_inverse = np.unique(state, return_inverse=True)
        unique_inverse = unique_inverse[len(self.unique):]
        num_novel_unique = len(unique) - len(self.unique)
        if num_novel_unique > 0:
            self._pad_count_matrix_novel_unique(num_novel_unique)
        self.count_matrix[idx, unique_inverse, self._state_idx] += 1
        self.unique = unique

    def _pad_count_matrix_novel_unique(self, num_novel_unique: int) -> None:
        """
        If we encounter novel unique values in state, we pad the count matrix
        with a set of axes for future count updates.
        """
        pad = np.zeros((self.num_envs, num_novel_unique, self._state_dim), dtype=np.uint16)
        if self.count_matrix is None:
            self.count_matrix = pad
        else:
            self.count_matrix = np.concatenate((self.count_matrix, pad), axis=1)

    def _mutual_info(self, i: int) -> float:
        """
        Compute mutual information content.
        """
        mutual_info = 0.
        for j in range(i):
            entropy_ij = self._entropy(self.count_matrix[i] + self.count_matrix[j])
            mutual_info += self.current_entropy[i] + self.current_entropy[j] - entropy_ij
        return mutual_info

    @staticmethod
    def _entropy(count_matrix: np.array) -> float:
        """
        Compute total trajectory information content.
        """
        return np.nan_to_num(entropy(count_matrix, base=2, axis=0).sum(), copy=False)

    def get_episode_data(self):
        """
        Generate episode data
        """
        episode_data = super().get_episode_data()
        episode_data.update({
            "Trajectory Information (bits)": self.current_entropy.tolist(),
            "Trajectory Mutual Information (bits)": self.current_mutual_info.tolist(),
        })
        return episode_data
