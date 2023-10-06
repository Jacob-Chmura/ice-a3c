import argparse
import os
import torch

def setup_experiment(args: argparse.Namespace) -> argparse.Namespace:
    """
    Set seeds and create directory for writing experiment logs, data, and model weights.
    """
    torch.manual_seed(args.seed)

    if args.experiment_name is None:
        args.experiment_name = _generate_experiment_name_from_args(args)
    args.Environment = _generate_environment_name_from_args(args)
    args.Method = _generate_method_description_from_args(args)
    args.experiment_dir = f"../experiments/{args.experiment_name}"
    args.result_dir = f"../results/{args.Environment}"
    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    return args

def is_gridworld_env(env_name: str) -> bool:
    return env_name.startswith("GW/")

def is_atari_env(env_name: str) -> bool:
    return env_name.startswith("ALE/")

def _generate_experiment_name_from_args(args: argparse.Namespace) -> str:
    """
    Generate an experiment name string based on key experiment arguments.
    """
    experiment_name = (
        f"{args.num_agents}-Agent-"
        f"{args.env_name.replace('/', '-')}-"
        f"Entropy_{args.entropy_coef}-"
        f"Mutual_info_{args.mutual_coef}-"
        f"Seed_{args.seed}"
    )
    return experiment_name

def _generate_environment_name_from_args(args: argparse.Namespace) -> str:
    """
    Generate an environment name string.
    """
    if is_gridworld_env(args.env_name):
        environment_name = f"{args.env_name.split('/')[1]} ({args.grid_size} x {args.grid_size})"
    elif is_atari_env(args.env_name):
        environment_name = args.env_name.split('/')[1]
    else:
        environment_name = "Unknown"
    return environment_name

def _generate_method_description_from_args(args: argparse.Namespace) -> str:
    """
    Generate a method description of the experiment based on experiment arguments.
    """
    if args.entropy_coef == 0 and args.mutual_coef == 0:
        method = "A3C"
    elif args.entropy_coef > 0 and args.mutual_coef == 0:
        method = "A3C with ICE"
    elif args.entropy_coef > 0 and args.mutual_coef > 0 and args.num_agents == 3:
        method = "A3C with MICE"
    elif args.entropy_coef > 0 and args.mutual_coef > 0 and args.num_agents == 5:
        method = "A3C with MICE-BIG"
    else:
        method = "Unknown"
    return method
