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


CHECKPOINT_FORMAT = "grounded-sac-v1"


def save_checkpoint(
    path: Path,
    *,
    learner: GroundedSACLearner,
    replay: VectorReplayBuffer,
    vector_step: int,
    generator: torch.Generator,
) -> None:
    """Atomically replace a complete, restartable learner/replay checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "learner": learner.state_dict(),
        "replay": replay.state_dict(),
        "vector_step": vector_step,
        "torch_rng_state": torch.get_rng_state(),
        "rollout_rng_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(
        f"checkpoint saved path={path} replay={len(replay)} updates={learner.update_count} "
        f"vector_step={vector_step}",
        flush=True,
    )


def load_checkpoint(
    path: Path,
    *,
    learner: GroundedSACLearner,
    replay: VectorReplayBuffer,
    generator: torch.Generator,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    learner.load_state_dict(payload["learner"])
    replay.load_state_dict(payload["replay"])
    torch.set_rng_state(payload["torch_rng_state"])
    generator.set_state(payload["rollout_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_states" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    vector_step = int(payload["vector_step"])
    print(
        f"checkpoint loaded path={path} replay={len(replay)} updates={learner.update_count} "
        f"vector_step={vector_step}",
        flush=True,
    )
    return vector_step


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
        batch = ReplayBatch(
            observation=self.obs.detach(), action=executed.detach(), reward=reward.squeeze(1).detach(),
            next_observation=next_obs.detach(), success=success.detach(), terminal=done.detach(),
            human_suffix=torch.zeros_like(done),
        )
        ids = self.replay.add_batch(batch)
        rows: list[dict[str, float]] = []
        for env_id in range(self.env.num_envs):
            raw_id = ids[env_id].item()
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
                "task/fallen": float(info["fallen"][env_id]),
                "task/recovery": float(info["recovery"][env_id]),
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
        if vector_steps % 100 == 0:
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
    parser.add_argument("--checkpoint-dir", default="checkpoints/grounded_sac")
    parser.add_argument("--checkpoint-every", type=int, default=5_000, help="learner updates between full-buffer checkpoints; 0 disables periodic saves")
    parser.add_argument("--resume", type=Path, default=None, help="resume learner/replay from a checkpoint")
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
    checkpoint_path = Path(args.checkpoint_dir) / "latest.pt"
    generator = torch.Generator(device=gs.device).manual_seed(args.seed + 1)
    vector_step = 0
    try:
        if args.resume is not None:
            vector_step = load_checkpoint(args.resume, learner=learner, replay=replay, generator=generator)
        else:
            collect_initial_demos(collector, args.demo_episodes)
            learner.fit_normalizer_once()
            print(f"starting IQL pretraining for {config.iql_pretrain_updates} updates", flush=True)
            for update in range(config.iql_pretrain_updates):
                metrics = learner._update_iql(replay.sample(config.batch_size, learner.device, human_suffix=True))
                if update % 100 == 0:
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
            save_checkpoint(checkpoint_path, learner=learner, replay=replay, vector_step=vector_step, generator=generator)

        update_budget = 0.0
        for step in range(vector_step, args.steps):
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
                if learner.update_count % 100 == 0:
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, learner.update_count)
                    writer.flush()
                    print(
                        f"update={learner.update_count} td_mse={metrics['sac/td_mse']:.5f} "
                        f"floor={metrics['floor/loss']:.5f} "
                        f"q_h350={metrics['sac/q_h350']:.5f}",
                        flush=True,
                    )
                if args.checkpoint_every and learner.update_count % args.checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, learner=learner, replay=replay, vector_step=step + 1, generator=generator)
                update_budget -= 1.0
            if step % 50 == 0:
                print(f"train step={step} replay={len(replay)} updates={learner.update_count}", flush=True)
            vector_step = step + 1
        save_checkpoint(checkpoint_path, learner=learner, replay=replay, vector_step=vector_step, generator=generator)
    except KeyboardInterrupt:
        save_checkpoint(checkpoint_path, learner=learner, replay=replay, vector_step=vector_step, generator=generator)
        raise
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
