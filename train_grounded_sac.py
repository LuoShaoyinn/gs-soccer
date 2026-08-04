"""Train no-fence human-grounded vector SAC on 512 floor-kick environments.

The pretrained ONNX teacher is used only to record the first 20 successful
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
        self.teacher_takeover = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
        self.human_suffix_limit: int | None = None
        self.successful_human_suffixes = 0

    def step(
        self,
        *,
        intervention_probability: float = 0.0,
        generator: torch.Generator | None = None,
        force_human: bool = False,
    ) -> list[dict[str, float]]:
        """Run one vector step with sticky fake-human intervention.

        An exploration episode independently triggers with probability `p` at
        each policy step.  The triggering action and all later actions through
        that environment's terminal transition are teacher actions.
        """

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
        if bool(human_mask.any()):
            teacher_action = self.teacher.act(self.env.get_state(self.env.all_envs_idx))
            executed = torch.where(human_mask[:, None], teacher_action, action)
        else:
            executed = action
        next_obs, reward, terminated, truncated, info = self.env.step(executed)
        done = (terminated | truncated).squeeze(1)
        success = info["success"].bool()
        batch = ReplayBatch(
            observation=self.obs.detach(), action=executed.detach(), reward=reward.squeeze(1).detach(),
            next_observation=next_obs.detach(), success=success.detach(), terminal=done.detach(),
            human_suffix=torch.zeros_like(done),
        )
        ids = self.replay.add_batch(batch)
        rows: list[dict[str, float]] = []
        for env_id, raw_id in enumerate(ids.tolist()):
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
    """Collect exactly 20 complete, successful, fully-human trajectories."""

    collector.human_suffix_limit = demo_episodes
    while collector.successful_human_suffixes < demo_episodes:
        collector.step(force_human=True)
    collector.human_suffix_limit = None
    count = int(collector.replay.human_suffix[:collector.replay.size].sum())
    print(f"Collected {demo_episodes} initial successful human demonstrations ({count} transitions).", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", default="refs/kick_ball_0625/20260624_144537_from20260624_111401/exported/actor.onnx")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--demo-episodes", type=int, default=20)
    parser.add_argument("--pretrain-updates", type=int, default=2_000)
    parser.add_argument("--updates-per-vector-step", type=float, default=10.0)
    parser.add_argument("--teacher-intervention-prob", type=float, default=0.0)
    parser.add_argument("--logdir", default="runs/grounded_sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if args.num_envs != 512:
        raise ValueError("this first training run is intentionally configured for 512 environments")
    torch.manual_seed(args.seed)
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    env = make_env(str(Path(args.actor)), args.num_envs, args.no_viewer)
    config = GroundedSACConfig(device=str(gs.device), iql_pretrain_updates=args.pretrain_updates)
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
        for update in range(config.iql_pretrain_updates):
            metrics = learner._update_iql(replay.sample(config.batch_size, learner.device, human_suffix=True))
            if update % 100 == 0:
                for key, value in metrics.items():
                    writer.add_scalar(key, value.detach().mean().item(), update)
                print(f"iql_pretrain update={update} td={metrics['iql/critic_td'].item():.5f}", flush=True)

        generator = torch.Generator(device=gs.device).manual_seed(args.seed + 1)
        update_budget = 0.0
        for step in range(args.steps):
            for episode_metric in collector.step(
                intervention_probability=args.teacher_intervention_prob,
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
                update_budget -= 1.0
            if step % 50 == 0:
                print(f"train step={step} replay={len(replay)} updates={learner.update_count}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
