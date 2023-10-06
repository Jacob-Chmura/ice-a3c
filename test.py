from algo import create_model_group
from util import Log
from util import Renderer

def run_test(args, done_training, env, shared_model_group):
    """
    Test process peforms inference and log evaluation metrics.
    """
    logger = Log(args)
    renderer = Renderer(args)
    local_model_group = create_model_group(args, shared_model_group)
    episode = 0

    while not done_training.is_set():
        actions = local_model_group.inference(env.states, is_train=False, greedy="GridWorld" not in args.env_name)
        env.step(actions)
        renderer(env.get_render_data())

        if env.is_done():
            episode_data = env.get_episode_data()
            episode_data["Frame Counts"] = local_model_group.get_frame_counts()
            logger(
                episode=episode,
                episode_data=episode_data,
            )

            env.reset()
            local_model_group.reset()
            local_model_group.reload_models(shared_model_group)
            local_model_group.save_models(args.experiment_dir)
            episode += 1

    renderer.render()
