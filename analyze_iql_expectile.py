"""Offline IQL expectile audit on a sharded successful-teacher replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from algorithm.grounded_sac import GroundedSACConfig, GroundedSACLearner, VectorReplayBuffer
from train_grounded_sac import _load_replay_snapshot


def factual_returns(replay: VectorReplayBuffer, gamma: float) -> torch.Tensor:
    """Monte Carlo returns for a compact replay of complete episode suffixes."""
    reward = replay.reward[: replay.size].detach().cpu()
    terminal = replay.terminal[: replay.size].detach().cpu()
    returns = torch.empty_like(reward)
    continuation = 0.0
    for index in range(replay.size - 1, -1, -1):
        if bool(terminal[index]):
            continuation = float(reward[index])
        else:
            continuation = float(reward[index]) + gamma * continuation
        returns[index] = continuation
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-buffer", type=Path, required=True)
    parser.add_argument("--expectile", type=float, default=0.99)
    parser.add_argument("--updates", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/iql_tau_099"))
    args = parser.parse_args()

    manifest = torch.load(args.teacher_buffer / "manifest.pt", map_location="cpu", weights_only=False)
    size = int(manifest["replay"]["size"])
    config = GroundedSACConfig(
        expectile=args.expectile, batch_size=args.batch_size,
        replay_capacity=size, device=args.device,
    )
    replay = VectorReplayBuffer(
        size, config.observation_dim, config.action_dim, device=args.device
    )
    _load_replay_snapshot(args.teacher_buffer, replay)
    learner = GroundedSACLearner(config, replay)
    learner.fit_normalizer_once()
    returns = factual_returns(replay, config.gamma)

    for update in range(1, args.updates + 1):
        batch = replay.sample(config.batch_size, learner.device, human_suffix=True)
        metrics = learner._update_iql(batch)
        if update == 1 or update % 500 == 0:
            print(
                f"update={update}/{args.updates} td={metrics['iql/critic_td'].item():.6f} "
                f"v={metrics['iql/expectile'].item():.6f} actor={metrics['iql/actor_loss'].item():.6f}",
                flush=True,
            )

    sample_count = min(65_536, replay.size)
    ids = torch.linspace(0, replay.size - 1, sample_count, device=replay.device).long()
    observation = replay.observation[ids]
    with torch.no_grad():
        action = learner.iql_action(observation)
        q350 = learner._iql_q_min(observation, action)[:, -1].cpu()
    target = returns[ids.cpu()]
    result = {
        "expectile": args.expectile,
        "updates": args.updates,
        "rows": replay.size,
        "factual_return_mean": float(target.mean()),
        "iql_q_h350_mean": float(q350.mean()),
        "q_minus_return_mean": float((q350 - target).mean()),
        "mae": float((q350 - target).abs().mean()),
        "correlation": float(torch.corrcoef(torch.stack((q350, target)))[0, 1]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(learner.iql_state_dict(), args.output / "iql.pt")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
