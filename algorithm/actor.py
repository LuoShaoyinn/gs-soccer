import os
import torch
from dataclasses import dataclass

import genesis as gs

from .algorithm import Algorithm

NUM_POLICY_JOINTS = 22

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


@dataclass(kw_only=True)
class ActorConfig:
    model_dir: str = "runs"
    model_file: str = "pi_plus_actor.pt"
    max_steps: int = 2500
    settle_steps: int = 50
    armature: tuple = (
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.01291, 0.01291, 0.01291,
    )
    damping: float = 0.0


class Actor(Algorithm):
    def __init__(self, env, cfg: ActorConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or ActorConfig()

        dev = gs.device
        self.idx = torch.tensor(POLICY_TO_GENESIS, dtype=torch.long, device=dev)
        policy_default_pos = torch.tensor(POLICY_DEFAULT_POS, dtype=torch.float32, device=dev)
        policy_action_scale = torch.tensor(POLICY_ACTION_SCALE, dtype=torch.float32, device=dev)
        self.default_pos = policy_default_pos[self.idx]
        self.action_scale = policy_action_scale[self.idx]

        self.net = ActorMLP().to(dev)
        ckpt = os.path.join(self.cfg.model_dir, self.cfg.model_file)
        self.net.load_state_dict(torch.load(ckpt, map_location=dev))
        self.net.eval()

        self._setup_env()

    def _setup_env(self):
        env = self.env
        dev = gs.device
        dofs_idx = env.robot.dofs_idx_local
        n = len(POLICY_TO_GENESIS)

        env.robot.robot.set_dofs_armature(
            torch.tensor(self.cfg.armature, dtype=torch.float32, device=dev),
            dofs_idx_local=dofs_idx,
        )
        env.robot.robot.set_dofs_damping(
            torch.full((n,), self.cfg.damping, dtype=torch.float32, device=dev),
            dofs_idx_local=dofs_idx,
        )

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def map_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.default_pos + action[:, self.idx] * self.action_scale

    def train(self):
        raise NotImplementedError("Sim2Sim actor does not support training")

    def eval(self):
        env = self.env
        N = env.num_envs
        dev = gs.device

        obs, info = env.reset()

        default_q = self.default_pos.unsqueeze(0).expand(N, -1)
        for _ in range(self.cfg.settle_steps):
            env.robot.step(default_q)
            env.gs_step()

        zeros = torch.zeros(N, NUM_POLICY_JOINTS, dtype=torch.float32, device=dev)
        obs, _, _, _, info = env.step(zeros)

        for step in range(self.cfg.max_steps):
            action = self.infer(obs)
            obs, reward, done, trunc, info = env.step(action)

            if step % 50 == 0:
                bp = info["body_pos"][0]
                bl = info["ball_pos"][0]
                bv = info["ball_vel"][0]
                cmd = info["soccer_cmd"][0]
                print(
                    f"  t={step:4d}  "
                    f"robot=[{bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}]  "
                    f"ball=[{bl[0]:.2f},{bl[1]:.2f}]  "
                    f"|v_ball|={bv.norm():.2f}  "
                    f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f},"
                    f"{cmd[3]:.2f},{cmd[4]:.2f},{cmd[5]:.2f},{cmd[6]:.2f}]"
                )

        print(f"\ndone. steps={self.cfg.max_steps}")
