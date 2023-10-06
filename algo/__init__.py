import argparse
from algo.gae import GAE
from algo.model.model import ActorCriticCNN
from algo.model.model_group import ModelGroup
from algo.shared_optimizers import SharedAdam

def create_model_group(args: argparse.Namespace, shared_model_groups=None) -> ModelGroup:
    """
    Create a group of actor critics. Sync weights with shared models if provided.
    """
    models = _create_models(args)
    optimizers = _create_optimizers(models, args)
    gaes = _create_gaes(args)
    model_group = ModelGroup(models, optimizers, gaes)
    if shared_model_groups is not None:
        model_group.reload_models(shared_model_groups)
    return model_group

def _create_models(args) -> list:
    """
    Instantiate the set of torch models
    """
    models = [
        ActorCriticCNN(num_inputs=args.grid_size, num_actions=args.num_actions, num_values=1)
        for _ in range(args.num_agents)
    ]
    return models

def _create_optimizers(models, args) -> list:
    """
    Instantiate the shared optimziers
    """
    optimizers = [
        SharedAdam(model.parameters(), args.lr)
        for model in models
    ]
    return optimizers

def _create_gaes(args) -> list:
    """
    Create generalized advantage estimation objects
    """
    gaes = [
        GAE(args.entropy_loss_coef, args.value_loss_coef, args.gamma, args.gae_lambda, args.max_grad_norm)
        for _ in range(args.num_agents)
    ]
    return gaes
