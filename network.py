# network.py
#   Defining networks
#

import torch
import torch.nn as nn
from typing import Mapping, Union, Any
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

def layer_init(layer, std=1.0, bias_const=0.0):
    """ Helper to perform Orthogonal Initialization """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, 
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, self.num_actions), std=0.01),
        )
        # Unitree default action noise std is 0.8.
        self.log_std_parameter = nn.Parameter(torch.log(torch.full((self.num_actions,), 0.8)))

    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = "") -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = "") -> tuple[torch.Tensor, dict]:
        return self.net(inputs["states"]), {}
