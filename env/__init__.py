import argparse
import gym
from env.wrappers import MiceWrapper
from env.wrappers import AtariObservationWrapper
from env.wrappers import GridWorldObservationWrapper
from env.custom_envs import ALL_ENVS_LIST
from env.custom_envs import ALL_GRID_SIZES
from env.custom_envs import ATARI_ENVS_LIST
from env.custom_envs import GRIDWORLD_ENVS_LIST

def create_env(args: argparse.Namespace) -> gym.Env:
    """
    Instantiate and return an environment group for the agent.
    """
    env = gym.vector.AsyncVectorEnv(
        [lambda: _create_env(args) for _ in range(args.num_agents)],
        shared_memory=True,
    )
    args.num_actions = env.single_action_space.n
    env = MiceWrapper(env, args)
    return env

def _create_env(args: argparse.Namespace) -> gym.Env:
    """
    Instantiate and return the appropriate environment for the agent.
    """
    if args.env_name in GRIDWORLD_ENVS_LIST:
        env = gym.make(args.env_name, grid_size=args.grid_size, seed=args.seed)
        env = GridWorldObservationWrapper(env)
    elif args.env_name in ATARI_ENVS_LIST:
        env = gym.make(args.env_name, render_mode="rgb_array")
        env = AtariObservationWrapper(env)
    else:
        raise ValueError(f"Provided environment {args.env_name} not configured.")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    return env
