import argparse
from pathlib import Path

import torch
import torch.nn as nn


def build_actor(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def main():
    parser = argparse.ArgumentParser(description="Export PI actor state checkpoint to TorchScript")
    parser.add_argument("--src", type=Path, default=Path("models_ckpt/pi_walk_40000.pt"))
    parser.add_argument("--dst", type=Path, default=Path("models_ckpt/pi_policy.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "pi_actor_state_v1":
        raise RuntimeError(f"Unsupported checkpoint format: {args.src}")

    actor = build_actor(checkpoint["dims"])
    actor.load_state_dict(checkpoint["state_dict"], strict=True)
    actor.eval()

    example = torch.zeros((1, checkpoint["dims"][0]), dtype=torch.float32)
    scripted = torch.jit.trace(actor, example)
    scripted.save(str(args.dst))
    print(f"Saved TorchScript policy: {args.dst}")


if __name__ == "__main__":
    main()
