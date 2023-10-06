import numpy as np
import gym
from abc import ABC
from abc import abstractmethod


class GridWorld(gym.Env, ABC):
    """
    Exploration grid world custom environment.
    """
    AGENT = 1
    WALL = 2

    def __init__(self, grid_size: int, seed: int):
        self.action_space = gym.spaces.Discrete(4)
        self.grid_size = grid_size
        self.observation = None
        self._action_to_direction = {
            0: np.array([ 1,  0], dtype=np.int16),
            1: np.array([ 0,  1], dtype=np.int16),
            2: np.array([-1,  0], dtype=np.int16),
            3: np.array([ 0, -1], dtype=np.int16),
        }
        self.seed(seed)
        self.reset()

    def reset(self) -> np.array:
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        self._reset_grid()
        self.steps = 0
        self.position = np.array([0, 0], dtype=np.uint16)
        self.path = [self.position[:]]
        return self.observation

    def step(self, action: int) -> tuple:
        """
        Run one timestep of environment dynamics.
        """
        new_position = self._get_new_position(action)
        self._update_state(new_position)
        self.path.append(self.position[:])
        self.steps += 1
        done = reward = self._get_reward()
        info = self._get_info()
        return self.observation, reward, done, info

    def seed(self, seed: int) -> None:
        """
        Sets the seed for this env's random number generator(s).
        """
        np.random.seed(seed)

    def _get_info(self) -> dict:
        """
        Create dictionary of environment info.
        """
        unique_states = set([self.grid_size * position[0] + position[1] for position in self.path])
        info = {
            "Trajectory Distinct States": len(unique_states),
            "Trajectory Visited States": self.path,
        }
        return info

    def _get_new_position(self, action: int) -> np.array:
        """
        Get the new agent position given action.
        """
        direction = self._action_to_direction[action]
        new_position = np.clip(self.position + direction, 0, self.grid_size - 1)
        return self._handle_collision(new_position)

    def _handle_collision(self, new_position: np.array) -> np.array:
        """
        Handle correction of position in case agent hits a wall.
        """
        if self.observation[tuple(new_position)] == GridWorld.WALL:
            new_position = self.position
        return new_position

    def _update_state(self, new_position: np.array) -> None:
        """
        Update the grid world state by marking the agent's position.
        """
        self.observation[tuple(self.position)] = 0
        self.observation[tuple(new_position)] = GridWorld.AGENT
        self.position = new_position

    @abstractmethod
    def _get_reward(self) -> float:
        """
        Get the environment reward.
        """

    @abstractmethod
    def _reset_grid(self) -> None:
        """
        Reset the grid.
        """


class GridWorldSparse(GridWorld):
    """
    Sparse grid world environment.
    """
    def _reset_grid(self) -> None:
        self.observation = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.observation[0,0] = GridWorld.AGENT

    def _get_reward(self) -> float:
        """
        Get the environment reward.
        """
        return 100 * all(self.position == self.grid_size - 1)


class GridWorldNoisy(GridWorld):
    """
    Exploration grid world custom environment with noisy tv.
    """
    NOISE_LOW = 3
    NOISE_HIGH = 6
    TV_SIZE_PERC = 0.125 # 12.5% of the grid is noise

    def __init__(self, grid_size: int, seed: int):
        self.TV_SIZE = int(self.TV_SIZE_PERC * grid_size)
        super().__init__(grid_size, seed)

    def _reset_grid(self) -> None:
        """
        Reset the grid and add noisy TV.
        """
        self.observation = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.observation[0,0] = GridWorld.AGENT
        self.observation[0:self.TV_SIZE:,-self.TV_SIZE:] = self.NOISE_LOW

    def _get_reward(self) -> float:
        """
        Get the environment reward.
        """
        return 100 * all(self.position == self.grid_size - 1)

    def _update_state(self, new_position: np.array) -> None:
        """
        Add noise to state space before marking agent's position on grid.
        """
        super()._update_state(new_position)
        self.observation[0:self.TV_SIZE:,-self.TV_SIZE:] = self._get_noise()

    def _get_noise(self) -> np.array:
        """
        Return TV noise
        """
        return np.random.randint(self.NOISE_LOW, self.NOISE_HIGH, size=(self.TV_SIZE, self.TV_SIZE))


class GridWorldWall(GridWorld):
    """
    Exploration grid world with wall and unobservable blocks.
    """
    def _reset_grid(self) -> None:
        self.observation = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.observation[0, 0] = GridWorld.AGENT
        self.observation[1:, 1:] = GridWorld.WALL

    def _get_reward(self) -> float:
        return 100 * all(self.position == [0, self.grid_size - 1])

    def _update_state(self, new_position) -> None:
        super()._update_state(new_position)
        self.observation[0, 1:] = 0 # keep the upper row unobservable by masking
