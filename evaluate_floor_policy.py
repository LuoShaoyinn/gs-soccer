"""Evaluate the frozen kick teacher or a grounded-SAC checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# `uv run` loads the root `.env`, where EGL is intentionally selected for
# headless training. Interactive Genesis uses an Xwayland/GLX window instead,
# so remove the override before importing Genesis or its rendering stack.
if "--viewer" in sys.argv:
    os.environ.pop("PYOPENGL_PLATFORM", None)

import numpy as np
import torch
import genesis as gs

from algorithm.grounded_sac import GroundedSACConfig
from algorithm.grounded_sac.networks import FrozenObservationNormalizer, VectorActor
from train_grounded_sac import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("teacher", "iql", "sac"), required=True)
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--actor", default="refs/kick_ball_0625/20260624_144537_from20260624_111401/exported/actor.onnx")
    parser.add_argument("--sac-checkpoint", type=Path)
    parser.add_argument("--iql-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    args = parser.parse_args()
    if args.policy == "sac" and (args.sac_checkpoint is None or args.iql_checkpoint is None):
        parser.error("SAC evaluation requires --sac-checkpoint and --iql-checkpoint")
    if args.policy == "iql" and args.iql_checkpoint is None:
        parser.error("IQL evaluation requires --iql-checkpoint")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    env = make_env(args.actor, args.num_envs, not args.viewer)
    observation, _ = env.reset()

    actor = normalizer = None
    if args.policy != "teacher":
        config = GroundedSACConfig(device=str(gs.device))
        actor = VectorActor(
            config.observation_dim, config.action_dim,
            config.hidden_dim, config.action_limit,
        ).to(gs.device).eval()
        normalizer = FrozenObservationNormalizer(config.observation_dim).to(gs.device).eval()
        iql_state = torch.load(args.iql_checkpoint, map_location=gs.device, weights_only=False)["state"]
        if args.policy == "sac":
            sac_state = torch.load(args.sac_checkpoint, map_location=gs.device, weights_only=False)["state"]
            actor.load_state_dict(sac_state["sac_actor"])
        else:
            actor.load_state_dict(iql_state["iql_actor"])
        normalizer.load_state_dict(iql_state["normalizer"])

    rows: list[tuple[float, float, float, float, float]] = []
    next_report = 1_000
    while len(rows) < args.episodes:
        with torch.no_grad():
            if args.policy == "teacher":
                action = env.MDP.policy_action(**env.get_state(env.all_envs_idx))
            else:
                assert actor is not None and normalizer is not None
                action = actor(normalizer(observation))
        observation, _, terminated, truncated, info = env.step(action)
        done = (terminated | truncated).squeeze(1)
        for index in torch.nonzero(done, as_tuple=False).squeeze(1).tolist():
            rows.append((
                float(info["success"][index]),
                float(info["fallen"][index]),
                float(info["timeout"][index]),
                float(info["step_count"][index]),
                float(info["episode_return"][index]),
            ))
        if len(rows) >= next_report:
            print(f"completed={len(rows)}/{args.episodes}", flush=True)
            next_report += 1_000

    env.close()
    values = np.asarray(rows[: args.episodes])
    result = {
        "policy": args.policy,
        "episodes": args.episodes,
        "success": float(values[:, 0].mean()),
        "fallen": float(values[:, 1].mean()),
        "timeout": float(values[:, 2].mean()),
        "episode_steps_mean": float(values[:, 3].mean()),
        "episode_steps_success_mean": float(values[values[:, 0] > 0.5, 3].mean()),
        "episode_return_mean": float(values[:, 4].mean()),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
