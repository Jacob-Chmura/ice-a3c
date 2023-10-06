import torch
import torch.nn.functional as F

class GAE:
    """
    Generalized advantage estimation.
    """
    def __init__(self, entropy_loss_coef: float, value_loss_coef: float, gamma: float, gae_lambda: float, max_grad_norm: float):
        self.entropy_loss_coef = entropy_loss_coef
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.gamma_gae_lambda = gae_lambda * gamma
        self.max_grad_norm = max_grad_norm
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def reset(self) -> None:
        """
        Reset all internal arrays.
        """
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def add(
        self,
        value: torch.Tensor,
        logit: torch.Tensor,
        prob: torch.Tensor,
        action: torch.Tensor,
        ) -> None:
        """
        Add data to internal arrays.
        """
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum()
        log_prob = log_prob.gather(0, action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def update(self, local_model, shared_model, optimizer) -> None:
        """
        Backpropogate through model.
        """
        if len(self.values) <= 1:
            return

        loss = self.get_loss()
        optimizer.zero_grad()
        loss.backward()
        self.ensure_shared_grads(local_model, shared_model)
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.max_grad_norm)
        optimizer.step()
        self.reset()

    def get_loss(self) -> torch.Tensor:
        """
        Compute the a3c loss.
        """
        value_loss, policy_loss, entropy_loss = 0, 0, 0
        target_v = self.values[-1].detach()
        gae = torch.zeros(1, 1)
        for t in reversed(range(len(self.rewards))):
            value_t, value_tp1 = self.values[t], self.values[t + 1]
            target_v = self.gamma * target_v + self.rewards[t]
            delta_t = self.rewards[t] + self.gamma * value_tp1 - value_t
            gae = gae * self.gamma_gae_lambda + delta_t
            value_loss += 0.5 * (target_v - value_t).pow(2)
            policy_loss -= self.log_probs[t] * gae.detach()
            entropy_loss -= self.entropies[t]
        return policy_loss + self.value_loss_coef * value_loss + self.entropy_loss_coef * entropy_loss

    @staticmethod
    def ensure_shared_grads(model, shared_model) -> None:
        """
        Copy gradients from local model to shared model in case it is None.
        """
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
