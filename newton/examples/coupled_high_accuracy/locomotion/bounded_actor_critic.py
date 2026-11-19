"""Tanh-bounded Gaussian actor compatible with rsl_rl 2.3.1."""

from __future__ import annotations

import torch
from torch.distributions import Normal


def install_bounded_actor_critic() -> None:
    """Expose the local policy to OnPolicyRunner's class-name resolver."""

    from rsl_rl.runners import on_policy_runner

    on_policy_runner.BoundedActorCritic = BoundedActorCritic


class BoundedActorCritic:
    """ActorCritic whose sampled and inferred actions are tanh bounded."""

    is_recurrent = False

    def __new__(cls, *args, **kwargs):
        from rsl_rl.modules import ActorCritic

        class _Implementation(ActorCritic):
            def update_distribution(self, observations):
                mean = self.actor(observations)
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                else:
                    std = torch.exp(self.log_std).expand_as(mean)
                self.distribution = Normal(mean, std)

            @property
            def action_mean(self):
                return self.distribution.mean

            @property
            def action_std(self):
                return self.distribution.stddev

            @property
            def entropy(self):
                return self.distribution.entropy().sum(dim=-1)

            def act(self, observations, **kwargs):
                self.update_distribution(observations)
                return torch.tanh(self.distribution.sample())

            def get_actions_log_prob(self, actions):
                eps = torch.finfo(actions.dtype).eps
                bounded = actions.clamp(-1.0 + eps, 1.0 - eps)
                pre_tanh = torch.atanh(bounded)
                log_prob = self.distribution.log_prob(pre_tanh)
                correction = torch.log1p(-bounded.square() + eps)
                return (log_prob - correction).sum(dim=-1)

            def act_inference(self, observations):
                return torch.tanh(self.actor(observations))

        instance = _Implementation(*args, **kwargs)
        instance.__class__.__name__ = cls.__name__
        return instance
