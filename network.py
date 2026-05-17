# network.py
#   Defining networks
#

import torch
import torch.nn as nn
from typing import Mapping, Union, Any
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def layer_init(layer, std=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 512), std=1.0),
            nn.LayerNorm(512),
            nn.ELU(),
            layer_init(nn.Linear(512, 256), std=1.0),
            nn.LayerNorm(256),
            nn.ELU(),
            layer_init(nn.Linear(256, self.num_actions), std=0.01),
            nn.Tanh(),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros((self.num_actions,)))

    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""):
        x = inputs.get("observations", inputs.get("states"))
        mean_actions = self.net(x)
        return mean_actions, {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 512), std=1.0),
            nn.LayerNorm(512),
            nn.ELU(),
            layer_init(nn.Linear(512, 256), std=1.0),
            nn.LayerNorm(256),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=0.5),
        )

    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""):
        x = inputs.get("observations", inputs.get("states"))
        v = self.net(x)
        return v, {}

