import torch
import torch.nn as nn
from typing import Mapping, Union, Any
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def layer_init(layer, std=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, min_log_std=-5, max_log_std=2)
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 256), std=1.0),
            nn.ELU(),
            layer_init(nn.Linear(256, 128), std=1.0),
            nn.ELU(),
            layer_init(nn.Linear(128, self.num_actions), std=0.01),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros((self.num_actions,)))

    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""):
        x = inputs.get("observations", inputs.get("states"))
        mean_actions = self.net(x)
        return mean_actions, {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 256), std=1.0),
            nn.ELU(),
            layer_init(nn.Linear(256, 128), std=1.0),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=0.5),
        )

    def compute(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""):
        x = inputs.get("observations", inputs.get("states"))
        return self.net(x), {}
