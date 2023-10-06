import gym
from util import is_atari_env
from util import is_gridworld_env

gym.envs.registration.register(
    id="GW/GridWorldSparse-v0",
    entry_point="env.custom_envs.gridworld:GridWorldSparse",
)

gym.envs.registration.register(
    id="GW/GridWorldNoisy-v0",
    entry_point="env.custom_envs.gridworld:GridWorldNoisy",
)

gym.envs.registration.register(
    id="GW/GridWorldWall-v0",
    entry_point="env.custom_envs.gridworld:GridWorldWall",
)

GRIDWORLD_ENVS_LIST = {
    env.id for env in gym.envs.registry.all() if is_gridworld_env(env.id)
}

ATARI_ENVS_LIST = {
    env.id for env in gym.envs.registry.all() if is_atari_env(env.id)
}

ALL_ENVS_LIST = sorted(ATARI_ENVS_LIST.union(GRIDWORLD_ENVS_LIST))
ALL_GRID_SIZES = [40, 80, 160]
