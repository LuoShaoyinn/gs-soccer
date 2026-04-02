from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.game import GameEnv, GameEnvConfig
from fields.soccer_field import SoccerField, SoccerFieldConfig
from models.game_model import GameModel, GameModelConfig
from models.pi_walk_model import PIWalkModel, PIWalkModelConfig
from policies import available_policy_modules, build_policy
from robots.controlled_robot import ControlledRobotWrapper, ControlledRobotWrapperConfig
from robots.pi import PI, PIConfig


PI_TARGET_Q_OFFSET = np.array(
    [
        -0.25,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.65,
        -1.2,
        0.65,
        -1.2,
        -0.4,
        -0.4,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def make_robot_wrapper(initial_pos: np.ndarray) -> ControlledRobotWrapperConfig:
    return ControlledRobotWrapperConfig(
        robot_cfg=PIConfig(initial_pos=initial_pos),
        robot_class=PI,
        ctrl_model_cfg=PIWalkModelConfig(n_dofs=20, target_q_offset=PI_TARGET_Q_OFFSET),
        ctrl_model_class=PIWalkModel,
        ctrl_policy_path="models_ckpt/pi_policy.pt",
    )


def make_env(
    *,
    num_envs: int,
    show_viewer: bool,
    policy_freq: int,
    sim_freq: int,
    ctrl_freq_ratio: int,
) -> GameEnv:
    return GameEnv(
        GameEnvConfig(
            robot_cfg={
                "red": make_robot_wrapper(np.array([-1.5, 0.0, 0.5], dtype=np.float32)),
                "blue": make_robot_wrapper(np.array([1.5, 0.0, 0.5], dtype=np.float32)),
            },
            robot_class={
                "red": ControlledRobotWrapper,
                "blue": ControlledRobotWrapper,
            },
            field_cfg=SoccerFieldConfig(),
            field_class=SoccerField,
            model_cfg=GameModelConfig(
                half_field_size=SoccerFieldConfig().half_field_size,
                goal_width=SoccerFieldConfig().goal_width,
            ),
            model_class=GameModel,
            num_envs=num_envs,
            show_viewer=show_viewer,
            env_spacing=0.0,
            policy_freq=policy_freq,
            sim_freq=sim_freq,
            ctrl_freq_ratio=ctrl_freq_ratio,
            max_collision_pairs=400,
            multiplier_collision_broad_phase=16,
            robot_reset_pos={
                "red": np.array([-1.5, 0.0, 0.0], dtype=np.float32),
                "blue": np.array([1.5, 0.0, np.pi], dtype=np.float32),
            },
        )
    )


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class Replay:
    obs: torch.Tensor
    act: torch.Tensor
    ptr: int
    full: bool


def create_replay(
    capacity: int, obs_dim: int, act_dim: int, device: torch.device
) -> Replay:
    return Replay(
        obs=torch.zeros((capacity, obs_dim), dtype=torch.float, device=device),
        act=torch.zeros((capacity, act_dim), dtype=torch.float, device=device),
        ptr=0,
        full=False,
    )


def replay_add(buf: Replay, obs: torch.Tensor, act: torch.Tensor) -> None:
    n = obs.shape[0]
    cap = buf.obs.shape[0]
    if n >= cap:
        buf.obs[:] = obs[-cap:]
        buf.act[:] = act[-cap:]
        buf.ptr = 0
        buf.full = True
        return

    end = buf.ptr + n
    if end <= cap:
        buf.obs[buf.ptr : end] = obs
        buf.act[buf.ptr : end] = act
    else:
        first = cap - buf.ptr
        buf.obs[buf.ptr :] = obs[:first]
        buf.act[buf.ptr :] = act[:first]
        rest = n - first
        buf.obs[:rest] = obs[first:]
        buf.act[:rest] = act[first:]
        buf.full = True
    buf.ptr = end % cap
    if buf.ptr == 0:
        buf.full = True


def replay_size(buf: Replay) -> int:
    return buf.obs.shape[0] if buf.full else buf.ptr


def replay_sample(buf: Replay, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    size = replay_size(buf)
    idx = torch.randint(0, size, (batch_size,), device=buf.obs.device)
    return buf.obs[idx], buf.act[idx]


def parse_args() -> argparse.Namespace:
    policy_modules = available_policy_modules()
    parser = argparse.ArgumentParser(description="Imitation learning for 1v1 dribble")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--headless", action="store_true", help="Disable viewer")
    parser.add_argument(
        "--num-envs", type=int, default=2048, help="Vectorized env count"
    )
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--policy-freq", type=int, default=50)
    parser.add_argument("--ctrl-freq-ratio", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--updates-per-iter", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--replay-size", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--experiment-name", type=str, default="dribble_1v1_il")
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument(
        "--expert-policy",
        choices=policy_modules,
        default="advanced_dribble",
        help="Expert policy module for imitation labels",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=policy_modules,
        default="zero",
        help="Opponent policy module",
    )
    parser.add_argument(
        "--team",
        choices=("red", "blue"),
        default="red",
        help="Which team to train",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path for eval or resume",
    )
    return parser.parse_args()


def evaluate(
    env: GameEnv,
    student: BCPolicy,
    opponent_policy,
    team: str,
    steps: int,
) -> None:
    other = "blue" if team == "red" else "red"
    obs, _ = env.reset()
    student.eval()
    with torch.no_grad():
        for _ in range(steps):
            team_act = student(obs[team])
            other_act = opponent_policy.act(obs[other])
            actions = {team: team_act, other: other_act}
            obs, _, _, _, _ = env.step(actions)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    eval_mode = args.eval
    env = make_env(
        num_envs=1 if eval_mode else args.num_envs,
        show_viewer=(not args.headless) if eval_mode else False,
        policy_freq=args.policy_freq,
        sim_freq=args.sim_freq,
        ctrl_freq_ratio=args.ctrl_freq_ratio,
    )

    obs_dim = env.observation_space[args.team].shape[0]
    act_dim = env.action_space[args.team].shape[0]

    student = BCPolicy(obs_dim=obs_dim, act_dim=act_dim).to(gs.device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=gs.device)
        student.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")

    opponent_policy = build_policy(args.opponent_policy, device=gs.device)

    if eval_mode:
        evaluate(env, student, opponent_policy, args.team, args.eval_steps)
        return

    expert_policy = build_policy(args.expert_policy, device=gs.device)
    other = "blue" if args.team == "red" else "red"

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    replay = create_replay(args.replay_size, obs_dim, act_dim, gs.device)
    obs, _ = env.reset()

    run_dir = Path("runs") / args.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for it in range(1, args.iterations + 1):
        student.eval()
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                expert_act = expert_policy.act(obs[args.team])
                student_act = student(obs[args.team])
                beta = max(0.0, 1.0 - it / max(1, int(0.7 * args.iterations)))
                mixed = beta * expert_act + (1.0 - beta) * student_act
                opp_act = opponent_policy.act(obs[other])
                actions = {args.team: mixed, other: opp_act}

                replay_add(replay, obs[args.team], expert_act)
                obs, _, _, _, _ = env.step(actions)

        if replay_size(replay) < args.batch_size:
            continue

        student.train()
        loss_acc = 0.0
        for _ in range(args.updates_per_iter):
            batch_obs, batch_act = replay_sample(replay, args.batch_size)
            pred = student(batch_obs)
            loss = F.mse_loss(pred, batch_act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            loss_acc += float(loss)

        loss_acc /= float(args.updates_per_iter)
        print(
            f"iter={it:04d} replay={replay_size(replay):6d} "
            f"beta={beta:.3f} loss={loss_acc:.6f}"
        )

        if it % args.save_interval == 0 or it == args.iterations:
            ckpt_path = ckpt_dir / f"iter_{it:04d}.pt"
            torch.save(student.state_dict(), ckpt_path)
            latest = ckpt_dir / "latest.pt"
            torch.save(student.state_dict(), latest)
            print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
