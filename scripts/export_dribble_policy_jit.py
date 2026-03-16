import argparse
from pathlib import Path

import torch
import torch.nn as nn


class DribblePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        # Kept for checkpoint compatibility with PPO policy state dicts.
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def _extract_policy_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dict")

    if "policy" in checkpoint and isinstance(checkpoint["policy"], dict):
        policy_state = checkpoint["policy"]
    else:
        policy_state = checkpoint

    if "net.0.weight" not in policy_state or "net.6.weight" not in policy_state:
        raise RuntimeError("Unsupported policy state dict format")
    return policy_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export dribble PPO policy checkpoint to TorchScript"
    )
    parser.add_argument("--src", type=Path, default=Path("runs/best_agent.pt"))
    parser.add_argument("--dst", type=Path, default=Path("runs/best_agent_policy.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location="cpu", weights_only=True)
    policy_state = _extract_policy_state(checkpoint)

    obs_dim = int(policy_state["net.0.weight"].shape[1])
    action_dim = int(policy_state["net.6.weight"].shape[0])

    policy = DribblePolicy(obs_dim=obs_dim, action_dim=action_dim)
    policy.load_state_dict(policy_state, strict=True)
    policy.eval()

    scripted = torch.jit.script(policy)
    scripted.save(str(args.dst))
    print(f"Saved TorchScript policy: {args.dst}")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")


if __name__ == "__main__":
    main()
