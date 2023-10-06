import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch.multiprocessing as mp
from env import ALL_ENVS_LIST
from env import ALL_GRID_SIZES
from env import create_env
from algo import create_model_group
from test import run_test
from train import run_train
from util import setup_experiment

parser = argparse.ArgumentParser(
    description="Multi-Agent Information Content Exploration",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--entropy-coef",      type=float, default=0.01,                       help="ICE intrinsic entropy reward coefficient.")
parser.add_argument("--entropy-loss-coef", type=float, default=0.01,                       help="Entropy loss weighting.")
parser.add_argument("--env-name",          type=str,   default="GW/GridWorldSparse-v0",    help="Environment to run.", choices=ALL_ENVS_LIST)
parser.add_argument("--experiment-name",   type=str,                                       help="Name of the experiment. A name is created if not provided.")
parser.add_argument("--gae-lambda",        type=float, default=1.00,                       help="Lambda in generalized advantage estimation.")
parser.add_argument("--gamma",             type=float, default=0.99,                       help="Discount factor.")
parser.add_argument("--grid-size",         type=int,   default=40,                         help="Size of grid if using GridWorldEnvironment.", choices=ALL_GRID_SIZES)
parser.add_argument("--load-model",        type=bool,  default=False,                      help="If true, source model weights from experiment spec.")
parser.add_argument("--lr",                type=float, default=0.0001,                     help="Learning rate.")
parser.add_argument("--max-episode-steps", type=int,   default=400,                        help="Number of steps per episode.")
parser.add_argument("--max-grad-norm",     type=float, default=50,                         help="Clip gradient norm on parameters.")
parser.add_argument("--max-episodes",      type=int,   default=200,                        help="Number of episodes per train process.")
parser.add_argument("--mutual-coef",       type=float, default=0.01,                       help="MICE intrinsic mutual information penalty coefficient.")
parser.add_argument("--num-agents",        type=int,   default=1,                          help="MICE number of agents to run.")
parser.add_argument("--num-processes",     type=int,   default=8,                          help="Number of asynchronous learners.")
parser.add_argument("--num-steps",         type=int,   default=20,                         help="Number of steps in generalized advantage estimation.")
parser.add_argument("--seed",              type=int,   default=0,                          help="RNG")
parser.add_argument("--value-loss-coef",   type=float, default=0.5,                        help="Value loss weighting.")

if __name__ == "__main__":
    args = setup_experiment(parser.parse_args())
    env = create_env(args)
    shared_model_group = create_model_group(args)
    if args.load_model:
        shared_model_group.load_models(args.experiment_dir)

    done_training = mp.Event()
    train_processes = []
    for _ in range(args.num_processes):
        p = mp.Process(target=run_train, args=(args, shared_model_group))
        p.start()
        train_processes.append(p)

    test_process = mp.Process(target=run_test, args=(args, done_training, env, shared_model_group))
    test_process.start()
    for p in train_processes:
        p.join()
    done_training.set()
    test_process.join()
