import torch
import torch.nn.functional as F
import numpy as np
from util import Counter

class ModelGroup:
    def __init__(self, models: list, optimizers: list, gaes: list):
        self.models = models
        self.optimizers = optimizers
        self.gaes = gaes
        self.num_agents = len(models)
        self._share_memory()
        self._init_hxcx()
        self._init_shared_frame_counts()

    def reload_models(self, shared_model_group: "ModelGroup") -> None:
        """
        Sync all model weights and frame counts with shared model group.
        """
        for idx in range(self.num_agents):
            self.models[idx].load_state_dict(shared_model_group.models[idx].state_dict())
            self.frame_counts[idx].set(shared_model_group.frame_counts[idx].get())

    def inference(self, states: np.array, dummy: bool=False, is_train: bool=True, greedy: bool=False) -> list:
        """
        Perform inference in model idx.
        """
        values, logits = self._forward(states, dummy, is_train)
        probs = [F.softmax(logit, dim=-1) for logit in logits]
        actions = [self.select_action(prob, greedy) for prob in probs]
        for gae, value, logit, prob, action in zip(self.gaes, values, logits, probs, actions):
            gae.add(value, logit, prob, action)
        actions = [action.item() for action in actions]
        return actions

    def terminal_inference(self, states: list, dones: list) -> None:
        """
        Perform inference for the last step in batch.
        """
        values, _ = self._forward(states, dummy=True)
        for gae, value, done in zip(self.gaes, values, dones):
            if done:
                values = torch.zeros(1)
            gae.values.append(value)

    def save_models(self, experiment_dir: str) -> None:
        """
        Save models to weight directory.
        """
        for idx, model in enumerate(self.models):
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "frame_count": self.frame_counts[idx].get(),
            }
            torch.save(checkpoint, f"{experiment_dir}/model{idx}.pth")

    def load_models(self, experiment_dir: str) -> None:
        """
        Load models from weight directory.
        """
        for idx, model in enumerate(self.models):
            checkpoint = torch.load(f"{experiment_dir}/model{idx}.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            self.frame_counts[idx].set(checkpoint["frame_count"])

    def add_reward(self, rewards: torch.Tensor) -> None:
        for gae, reward in zip(self.gaes, rewards):
            gae.rewards.append(reward)

    def update(self, shared_model_group: "ModelGroup") -> None:
        for idx in range(self.num_agents):
            self.gaes[idx].update(
                self.models[idx],
                shared_model_group.models[idx],
                shared_model_group.optimizers[idx]
            )
            shared_model_group.frame_counts[idx].set(self.frame_counts[idx].get())
        self._detach_hxcx()

    def reset(self) -> None:
        for gae in self.gaes:
            gae.reset()
        self._init_hxcx()

    def get_frame_counts(self) -> list:
        return [counter.get() for counter in self.frame_counts]

    def _forward(self, states: np.array, dummy: bool=False, is_train: bool=True) -> tuple:
        """
        Perform inference in model idx.
        """
        states = torch.from_numpy(np.expand_dims(states, (1, 2)))
        values, logits = [], []
        for idx, state in enumerate(states):
            with torch.set_grad_enabled(is_train):
                if dummy:
                    value, logit, (_, _) = self.models[idx](state, self.hxs[idx], self.cxs[idx])
                else:
                    value, logit, (self.hxs[idx], self.cxs[idx]) = self.models[idx](state, self.hxs[idx], self.cxs[idx])
            values.append(value.squeeze(0))
            logits.append(logit.squeeze(0))
            self.frame_counts[idx].increment(increment_value=state.shape[0])
        return values, logits

    def _share_memory(self) -> None:
        """
        Put the model parameters and loss information for each model into shared memory.
        """
        for model, optimizer in zip(self.models, self.optimizers):
            model.share_memory()
            optimizer.share_memory()

    def _init_shared_frame_counts(self) -> None:
        """
        Initialize process-safe counter for processed frames for each model.
        """
        self.frame_counts = [Counter() for _ in range(self.num_agents)]

    def _init_hxcx(self) -> None:
        """
        Initialize lstm states.
        """
        self.hxs = [torch.zeros(1, 256) for _ in range(self.num_agents)]
        self.cxs = [torch.zeros(1, 256) for _ in range(self.num_agents)]

    def _detach_hxcx(self):
        """
        Detach lstm states of model at idx.
        """
        self.hxs = [hx.detach() for hx in self.hxs]
        self.cxs = [cx.detach() for cx in self.cxs]

    @staticmethod
    def select_action(prob: torch.Tensor, greedy: bool = False):
        """
        Select discrete action from probability tensor.
        Sample according to probabilities unless greedy is true, in which case the max probability
        action is selected.
        """
        if greedy:
            action = prob.max(0, keepdim=True)[1].detach()
        else:
            action = prob.multinomial(num_samples=1).detach()
        return action
