import torch
import torch.nn as nn
from torch.distributions import Normal
from skrl.models.torch import DeterministicMixin, Model


class Policy(Model):
    def __init__(self, observation_space, action_space, device, hidden_dims=(512, 256, 128)):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        self._log_std_min = -5.0
        self._log_std_max = 2.0
        layers = []
        prev = self.num_observations
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ELU()]
            prev = hidden
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev, self.num_actions)
        nn.init.uniform_(self.mean_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_layer.bias)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.2))

    def compute(self, inputs, role=""):
        h = self.net(inputs["observations"])
        mean = self.mean_layer(h)
        log_std = self.log_std_parameter.expand_as(mean).clamp(self._log_std_min, self._log_std_max)
        return mean, {"log_std": log_std}

    def act(self, inputs, role=""):
        mean, outputs = self.compute(inputs, role)
        dist = Normal(mean, outputs["log_std"].exp())
        action = dist.rsample()
        outputs["log_prob"] = dist.log_prob(action).sum(-1, keepdim=True)
        return action, outputs


class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_dims=(512, 256, 128)):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        layers = []
        prev = self.num_observations + self.num_actions
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ELU()]
            prev = hidden
        head = nn.Linear(prev, 1)
        nn.init.uniform_(head.weight, -1e-3, 1e-3)
        nn.init.zeros_(head.bias)
        layers += [head]
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        q = self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1))
        return q, {}

