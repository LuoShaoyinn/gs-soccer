import os
import torch

import genesis as gs

NUM_POLICY_JOINTS = 22
NUM_GENESIS_JOINTS = 20

POLICY_TO_GENESIS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

POLICY_DEFAULT_POS = [
    0.0, -0.25, 0.0, -0.25, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.65, 0.0, 0.65, 0.0,
    -0.4, -0.4, 0.0, 0.0,
]

POLICY_ACTION_SCALE = [
    0.096, 0.098, 0.154, 0.098, 0.154, 0.096,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.098, 0.098, 0.098,
]


class ActorMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(632, 512)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 22)
        self.act = torch.nn.ELU()

    def forward(self, x):
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class Actor:
    def __init__(self, model_dir: str = "runs", model_file: str = "pi_plus_actor.pt"):
        self.net = ActorMLP().to(gs.device)
        ckpt = os.path.join(model_dir, model_file)
        self.net.load_state_dict(torch.load(ckpt, map_location=gs.device))
        self.net.eval()

        dev = gs.device
        self.idx = torch.tensor(POLICY_TO_GENESIS, dtype=torch.long, device=dev)
        self.default_pos = torch.tensor(POLICY_DEFAULT_POS, dtype=torch.float32, device=dev)
        self.action_scale = torch.tensor(POLICY_ACTION_SCALE, dtype=torch.float32, device=dev)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def map_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.default_pos[self.idx] + action[:, self.idx] * self.action_scale[self.idx]
