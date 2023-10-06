from algo import create_model_group
from env import create_env

def run_train(args, shared_model_group):
    """
    Train processes asynchronously run actor critic training.
    """
    env = create_env(args)
    local_model_group = create_model_group(args, shared_model_group)
    episode = 0

    while episode < args.max_episodes:
        for step in range(args.num_steps):
            actions = local_model_group.inference(env.states)
            rewards = env.step(actions)
            local_model_group.add_reward(rewards)

            if (step == args.num_steps - 1) or env.is_done():
                local_model_group.terminal_inference(env.states, env.dones)
                local_model_group.update(shared_model_group)
                local_model_group.reload_models(shared_model_group)

            if env.is_done():
                env.reset()
                local_model_group.reset()
                episode += 1
