"""Train no-fence human-grounded vector SAC on 512 floor-kick environments.

The pretrained teacher is used only to record the initial successful
human-equivalent demonstrations (and optionally to simulate later human
intervention suffixes).  It is never an SAC imitation objective.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gs-soccer-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gs-soccer-matplotlib")

import genesis as gs

from algorithm import TeacherActor
from algorithm.grounded_sac import GroundedSACConfig, GroundedSACLearner, VectorReplayBuffer
from algorithm.grounded_sac.replay import ReplayBatch
from envs import Env, EnvConfig
from fields import BallField, BallFieldConfig
from MDPs import FloorIQLConfig, FloorIQLMDP
from robots import KickPI, KickPIConfig


def make_env(actor_path: str, num_envs: int, no_viewer: bool) -> Env:
    robot_cfg = KickPIConfig()
    field_cfg = BallFieldConfig(
        ball_radius=0.07, ball_mass=0.25, ball_damping=0.05,
        ball_friction=0.6, ball_init_pos=np.array([1.0, 0.0, 0.07], dtype=np.float32),
    )
    cfg = EnvConfig(
        robot_cfg=robot_cfg, robot_class=KickPI,
        field_cfg=field_cfg, field_class=BallField,
        MDP_cfg=FloorIQLConfig(actor_path=actor_path), MDP_class=FloorIQLMDP,
        policy_freq=50, sim_freq=500, num_envs=num_envs, show_viewer=not no_viewer,
    )
    return Env(cfg)


class Collector:
    def __init__(self, env: Env, replay: VectorReplayBuffer, learner: GroundedSACLearner, teacher: TeacherActor) -> None:
        self.env, self.replay, self.learner, self.teacher = env, replay, learner, teacher
        self.obs, _ = env.reset()
        self.episodes: list[list[tuple[int, bool]]] = [[] for _ in range(env.num_envs)]
        self.completed = 0
        self.invalid_transitions = 0
        self.teacher_takeover = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
        self.human_suffix_limit: int | None = None
        self.successful_human_suffixes = 0

    def step(
        self,
        *,
        intervention_probability: float = 0.0,
        exploration_std: float = 0.0,
        generator: torch.Generator | None = None,
        force_human: bool = False,
    ) -> list[dict[str, float]]:
        """Run one vector step with sticky fake-human intervention.

        An exploration episode independently triggers with probability `p` at
        each policy step.  The triggering action and all later actions through
        that environment's terminal transition are teacher actions.
        """

        # A failed physics state must never be fed through the policy or into
        # replay. Reset it before proposing an action.
        invalid_state = ~torch.isfinite(self.obs).all(dim=1)
        if invalid_state.any():
            invalid_idx = torch.nonzero(invalid_state).squeeze(1)
            reset_obs, _ = self.env.reset(invalid_idx)
            self.obs[invalid_idx] = reset_obs
            for env_id in invalid_idx.tolist():
                self.episodes[env_id] = []
            self.teacher_takeover[invalid_idx] = False
            self.invalid_transitions += len(invalid_idx)
            print(f"nan_guard reset {len(invalid_idx)} non-finite pre-action environments; total={self.invalid_transitions}", flush=True)
        if force_human:
            human_mask = torch.ones_like(self.teacher_takeover)
        else:
            trigger = (
                torch.rand(self.env.num_envs, generator=generator, device=gs.device)
                < intervention_probability
            ) & ~self.teacher_takeover
            self.teacher_takeover |= trigger
            human_mask = self.teacher_takeover
        action, _ = self.learner.controller_action(self.obs)
        if exploration_std > 0.0 and not force_human:
            noise = torch.randn(
                action.shape, generator=generator, device=action.device,
                dtype=action.dtype,
            ) * exploration_std
            action = (action + noise).clamp(-1.0, 1.0)
        if bool(human_mask.any()):
            teacher_action = self.teacher.act(self.env.get_state(self.env.all_envs_idx))
            executed = torch.where(human_mask[:, None], teacher_action, action)
        else:
            executed = action
        next_obs, reward, terminated, truncated, info = self.env.step(executed)
        done = (terminated | truncated).squeeze(1)
        success = info["success"].bool()
        valid = (
            torch.isfinite(self.obs).all(dim=1)
            & torch.isfinite(executed).all(dim=1)
            & torch.isfinite(reward.squeeze(1))
            & torch.isfinite(next_obs).all(dim=1)
        )
        if (~valid).any():
            invalid_idx = torch.nonzero(~valid).squeeze(1)
            raw_state = self.env.get_state(self.env.all_envs_idx)
            state_bad = {
                name: int((~torch.isfinite(value[invalid_idx])).sum().item())
                for name, value in raw_state.items()
                if value.is_floating_point() and not bool(torch.isfinite(value[invalid_idx]).all())
            }
            bad_parts = []
            for name, value in {
                "obs": self.obs,
                "action": executed,
                "reward": reward.squeeze(1),
                "next_obs": next_obs,
            }.items():
                count = int((~torch.isfinite(value[invalid_idx])).sum().item())
                if count:
                    bad_parts.append(f"{name}={count}")
            reset_obs, _ = self.env.reset(invalid_idx)
            next_obs = next_obs.clone()
            next_obs[invalid_idx] = reset_obs
            self.teacher_takeover[invalid_idx] = False
            for env_id in invalid_idx.tolist():
                self.episodes[env_id] = []
            self.invalid_transitions += len(invalid_idx)
            print(
                f"nan_guard dropped envs={invalid_idx.tolist()} "
                f"({' '.join(bad_parts)}) state_nonfinite={state_bad} "
                f"total={self.invalid_transitions}",
                flush=True,
            )
        batch = ReplayBatch(
            observation=self.obs.detach(), action=executed.detach(), reward=reward.squeeze(1).detach(),
            next_observation=next_obs.detach(), success=success.detach(), terminal=done.detach(),
            human_suffix=torch.zeros_like(done),
        )
        valid_idx = torch.nonzero(valid).squeeze(1)
        stored = ReplayBatch(**{
            name: getattr(batch, name)[valid_idx]
            for name in ReplayBatch.__dataclass_fields__
        })
        ids = self.replay.add_batch(stored)
        row_ids = dict(zip(valid_idx.tolist(), ids.tolist(), strict=True))
        rows: list[dict[str, float]] = []
        for env_id in range(self.env.num_envs):
            if not bool(valid[env_id]):
                continue
            raw_id = row_ids[env_id]
            self.episodes[env_id].append((raw_id, bool(human_mask[env_id])))
            if not bool(done[env_id]):
                continue
            trajectory = self.episodes[env_id]
            suffix: list[int] = []
            for transition_id, intervened in reversed(trajectory):
                if not intervened:
                    break
                suffix.append(transition_id)
            can_add_suffix = (
                self.human_suffix_limit is None
                or self.successful_human_suffixes < self.human_suffix_limit
            )
            if bool(success[env_id]) and suffix and can_add_suffix:
                self.replay.mark_human_suffix(suffix)
                self.successful_human_suffixes += 1
            rows.append({
                "task/success": float(success[env_id]),
                "task/episode_steps": float(len(trajectory)),
                "task/intervention_fraction": float(sum(flag for _, flag in trajectory) / len(trajectory)),
                "task/sac_control_fraction": float(sum(not flag for _, flag in trajectory) / len(trajectory)),
                "task/iql_control_fraction": 0.0,  # fence is deliberately disabled
            })
            self.episodes[env_id] = []
            self.teacher_takeover[env_id] = False
            self.completed += 1
        self.obs = next_obs
        return rows


def collect_initial_demos(collector: Collector, demo_episodes: int) -> None:
    """Collect the requested complete, successful, fully-human trajectories."""

    collector.human_suffix_limit = demo_episodes
    vector_steps = 0
    while collector.successful_human_suffixes < demo_episodes:
        collector.step(force_human=True)
        vector_steps += 1
        if vector_steps % 25 == 0:
            print(
                f"collect_demos step={vector_steps} successes="
                f"{collector.successful_human_suffixes}/{demo_episodes}",
                flush=True,
            )
    collector.human_suffix_limit = None
    count = int(collector.replay.human_suffix[:collector.replay.size].sum())
    print(f"Collected {demo_episodes} initial successful human demonstrations ({count} transitions).", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", default="refs/kick_ball_0625/20260624_144537_from20260624_111401/exported/actor.onnx")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--demo-episodes", type=int, default=2_000)
    parser.add_argument("--pretrain-updates", type=int, default=2_000)
    parser.add_argument("--updates-per-vector-step", type=float, default=1.0)
    parser.add_argument("--exploration-std", type=float, default=0.05)
    parser.add_argument("--teacher-intervention-prob", type=float, default=0.0)
    parser.add_argument("--logdir", default="runs/grounded_sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if args.num_envs != 512:
        raise ValueError("this first training run is intentionally configured for 512 environments")
    torch.manual_seed(args.seed)
    print("initializing Genesis scene (512 environments)...", flush=True)
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    env = make_env(str(Path(args.actor)), args.num_envs, args.no_viewer)
    print(f"Genesis scene ready on {gs.device}; allocating replay...", flush=True)
    config = GroundedSACConfig(device=str(gs.device), iql_pretrain_updates=args.pretrain_updates)
    config.exploration_std = args.exploration_std
    utd = args.updates_per_vector_step * config.batch_size / args.num_envs
    print(
        f"batch_size={config.batch_size} updates_per_vector_step={args.updates_per_vector_step:g} "
        f"effective_UTD={utd:g} exploration_std={config.exploration_std:g}",
        flush=True,
    )
    replay = VectorReplayBuffer(
        config.replay_capacity, config.observation_dim, config.action_dim,
        device=gs.device,
    )
    learner = GroundedSACLearner(config, replay)
    collector = Collector(env, replay, learner, TeacherActor(env))
    writer = SummaryWriter(args.logdir)
    try:
        collect_initial_demos(collector, args.demo_episodes)
        learner.fit_normalizer_once()
        print(f"starting IQL pretraining for {config.iql_pretrain_updates} updates", flush=True)
        for update in range(config.iql_pretrain_updates):
            metrics = learner._update_iql(replay.sample(config.batch_size, learner.device, human_suffix=True))
            if update % 10 == 0:
                for key, value in metrics.items():
                    writer.add_scalar(key, value.detach().mean().item(), update)
                writer.flush()
                print(
                    f"iql_pretrain update={update}/{config.iql_pretrain_updates} "
                    f"td={metrics['iql/critic_td'].item():.5f} "
                    f"expectile={metrics['iql/expectile'].item():.5f} "
                    f"actor={metrics['iql/actor_loss'].item():.5f}",
                    flush=True,
                )

        generator = torch.Generator(device=gs.device).manual_seed(args.seed + 1)
        update_budget = 0.0
        for step in range(args.steps):
            for episode_metric in collector.step(
                intervention_probability=args.teacher_intervention_prob,
                exploration_std=config.exploration_std,
                generator=generator,
            ):
                for key, value in episode_metric.items():
                    writer.add_scalar(key, value, step * args.num_envs)
            update_budget += args.updates_per_vector_step
            while update_budget >= 1.0:
                metrics = learner.update()
                if learner.update_count % 10 == 0:
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, learner.update_count)
                    writer.flush()
                    print(
                        f"update={learner.update_count} td_mse={metrics['sac/td_mse']:.5f} "
                        f"floor={metrics['floor/loss']:.5f} "
                        f"q_h350={metrics['sac/q_h350']:.5f}",
                        flush=True,
                    )
                update_budget -= 1.0
            if step % 50 == 0:
                print(f"train step={step} replay={len(replay)} updates={learner.update_count}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
