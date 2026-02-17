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
            nn.LayerNorm(512), # Keeps inner activations in check
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256),
            nn.ELU(),
            layer_init(nn.Linear(256, self.num_actions), std=0.01), # Small std for final layer
            nn.Tanh() # Strictly clips the output mean to [-1, 1]
        )
        # Log std as a parameter (initialized to 0 results in std=1.0)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    @torch.compile()
    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = "") -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 512)),
            nn.LayerNorm(512),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0) # Value isn't clipped; needs to represent any range
        )

    @torch.compile()
    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = "") -> tuple[torch.Tensor, dict]:
        return self.net(inputs["states"]), {}
