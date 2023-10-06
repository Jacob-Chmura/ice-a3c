import imageio
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import deque
from util.constants import ATARI_ACTION_MAPPING
from util.constants import GRIDWORLD_ACTION_MAPPING
from util.setup import is_atari_env
from util.setup import is_gridworld_env


class Renderer:
    """
    Renderer utility class. Processes episode frames
    and serializes to file for replay.
    """
    def __init__(self, args):
        self.env_name = args.env_name
        self.num_agents = args.num_agents
        self.buffer_size = args.max_episode_steps
        self.render_file_name = [
            f"{args.result_dir}/{args.Method}_seed_{args.seed}_agent_{i}_replay.gif" for i in range(self.num_agents)
        ]
        self.frames = [deque([], maxlen=self.buffer_size) for _ in range(self.num_agents)]
        self.annotations = [deque([], maxlen=self.buffer_size) for _ in range(self.num_agents)]

    def __call__(self, render_data: dict) -> None:
        """
        Called during each test episode step.
        Process frames from environment and render gif to disc when queues are full.
        """
        for i in range(self.num_agents):
            frame, annotation = self._process_agent_frame(
                render_data["State"][i],
                render_data["Action"][i],
                render_data["Step"][i]
            )
            self.frames[i].append(frame)
            self.annotations[i].append(annotation)

    def _process_agent_frame(self, state: np.ndarray, action: int, step: int) -> tuple:
        frame = self._process_state(state)
        action = self._process_action(action)
        annotation = f"Step: {step} Action: {action}"
        return frame, annotation

    def _process_state(self, state: np.ndarray, upsample=8) -> np.ndarray:
        state = (state * 256).astype(np.uint8)
        state = state.repeat(upsample, axis=0).repeat(upsample, axis=1)
        return state

    def _process_action(self, action: int) -> str:
        if is_gridworld_env(self.env_name):
            action = GRIDWORLD_ACTION_MAPPING[action]
        elif is_atari_env(self.env_name):
            action = ATARI_ACTION_MAPPING[action]
        else:
            action = str(action)
        return action

    def render(self) -> None:
        for i in range(self.num_agents):
            annotated_frames = []
            for j, frame in enumerate(self.frames[i]):
                f, ax = plt.subplots(figsize=(12, 12), dpi=50)
                ax.imshow(frame)
                ax.text(5, 5, self.annotations[i][j], fontsize="xx-large", va="top", color="red")
                ax.set_axis_off()
                ax.set_position([0, 0, 1, 1])
                f.canvas.draw()
                annotated_frames.append(np.asarray(f.canvas.renderer.buffer_rgba()))
                plt.close(f)
            imageio.mimsave(self.render_file_name[i], annotated_frames, fps=5)
