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


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_networks(
    run_dir: Path,
    *,
    learner: GroundedSACLearner,
    vector_step: int,
    generator: torch.Generator,
) -> None:
    """Save separate IQL and SAC snapshots, matching the reference layout."""
    common: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "vector_step": vector_step,
        "torch_rng_state": torch.get_rng_state(),
        "rollout_rng_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        common["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    update = learner.update_count
    _atomic_torch_save({**common, "state": learner.iql_state_dict()}, run_dir / "iql" / f"model_{update}.pt")
    _atomic_torch_save({**common, "state": learner.sac_state_dict()}, run_dir / "sac" / f"model_{update}.pt")
    print(
        f"networks saved update={update} vector_step={vector_step}",
        flush=True,
    )


def save_buffer(
    run_dir: Path,
    *,
    learner: GroundedSACLearner,
    replay: VectorReplayBuffer,
    vector_step: int,
) -> Path:
    """Save the complete populated replay separately from network snapshots."""
    path = run_dir / "buffer" / f"buffer_{vector_step}.pt"
    _atomic_torch_save({
        "format": CHECKPOINT_FORMAT,
        "vector_step": vector_step,
        "learner_update_count": learner.update_count,
        "replay": replay.state_dict(),
    }, path)
    print(f"buffer saved path={path} replay={len(replay)}", flush=True)
    return path


def load_run(
    run_dir: Path,
    *,
    learner: GroundedSACLearner,
    replay: VectorReplayBuffer,
    generator: torch.Generator,
) -> int:
    candidates = []
    for path in (run_dir / "buffer").glob("buffer_*.pt"):
        suffix = path.stem.removeprefix("buffer_")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"no replay checkpoint under {run_dir / 'buffer'}")
    vector_step, buffer_path = max(candidates)
    buffer_payload = torch.load(buffer_path, map_location="cpu", weights_only=False)
    if buffer_payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported replay checkpoint format in {buffer_path}")
    update = int(buffer_payload["learner_update_count"])
    iql_path = run_dir / "iql" / f"model_{update}.pt"
    sac_path = run_dir / "sac" / f"model_{update}.pt"
    if not iql_path.exists() or not sac_path.exists():
        raise FileNotFoundError(f"missing network pair for replay checkpoint update {update}")
    iql_payload = torch.load(iql_path, map_location="cpu", weights_only=False)
    sac_payload = torch.load(sac_path, map_location="cpu", weights_only=False)
    learner.load_iql_state_dict(iql_payload["state"])
    learner.load_sac_state_dict(sac_payload["state"])
    replay.load_state_dict(buffer_payload["replay"])
    torch.set_rng_state(sac_payload["torch_rng_state"])
    generator.set_state(sac_payload["rollout_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_states" in sac_payload:
        torch.cuda.set_rng_state_all(sac_payload["cuda_rng_states"])
    print(
        f"run loaded path={run_dir} replay={len(replay)} updates={learner.update_count} "
        f"vector_step={vector_step}",
        flush=True,
    )
    return vector_step


def restore_human_buffer(run_dir: Path, replay: VectorReplayBuffer) -> int:
    """Load only successful human suffixes from a prior run's latest buffer."""
    candidates = []
    for path in (run_dir / "buffer").glob("buffer_*.pt"):
        suffix = path.stem.removeprefix("buffer_")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"no replay checkpoint under {run_dir / 'buffer'}")
    _, buffer_path = max(candidates)
    payload = torch.load(buffer_path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported replay checkpoint format in {buffer_path}")
    replay.load_state_dict(payload["replay"])
    count = replay.retain_human_suffix()
    # This is a new experiment seeded by imported demonstrations, so report
    # collection totals relative to the restored dataset rather than its
    # source experiment's discarded transitions.
    replay.total_transitions = count
    print(f"restored successful human buffer path={buffer_path} transitions={count}", flush=True)
    return count


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
    """Store only complete successful teacher trajectories in replay.

    Failed teacher attempts are staged only until their terminal step, then
    discarded.  This mirrors the fixed successful-demonstration dataset and
    prevents unsuccessful collection attempts from consuming replay capacity.
    """

    def cpu_row(batch: ReplayBatch, env_id: int) -> ReplayBatch:
        return ReplayBatch(**{
            name: getattr(batch, name)[env_id : env_id + 1].detach().cpu().clone()
            for name in ReplayBatch.__dataclass_fields__
        })

    collector.episodes = [[] for _ in range(collector.env.num_envs)]
    collector.teacher_takeover.zero_()
    vector_steps = 0
    while collector.successful_human_suffixes < demo_episodes:
        state = collector.env.get_state(collector.env.all_envs_idx)
        action = collector.teacher.act(state)
        next_obs, reward, terminated, truncated, info = collector.env.step(action)
        done = (terminated | truncated).squeeze(1)
        success = info["success"].bool()
        batch = ReplayBatch(
            observation=collector.obs, action=action, reward=reward.squeeze(1),
            next_observation=next_obs, success=success, terminal=done,
            human_suffix=torch.zeros_like(done),
        )
        for env_id in range(collector.env.num_envs):
            collector.episodes[env_id].append(cpu_row(batch, env_id))
            if not bool(done[env_id]):
                continue
            episode = collector.episodes[env_id]
            if bool(success[env_id]) and collector.successful_human_suffixes < demo_episodes:
                successful_batch = ReplayBatch(**{
                    name: torch.cat([getattr(row, name) for row in episode], dim=0)
                    for name in ReplayBatch.__dataclass_fields__
                })
                ids = collector.replay.add_batch(successful_batch)
                collector.replay.mark_human_suffix(ids.tolist())
                collector.successful_human_suffixes += 1
            collector.episodes[env_id] = []
            collector.completed += 1
        collector.obs = next_obs
        vector_steps += 1
        if vector_steps % 100 == 0:
            print(
                f"collect_demos step={vector_steps} successes="
                f"{collector.successful_human_suffixes}/{demo_episodes}",
                flush=True,
            )
    count = len(collector.replay)
    print(f"Collected {demo_episodes} initial successful human demonstrations ({count} transitions).", flush=True)
    # The success target can be reached while other vector environments still
    # hold staged demo rows.  Start SAC from fresh episode boundaries so its
    # normal `(replay_id, intervened)` bookkeeping cannot inherit that staging.
    collector.obs, _ = collector.env.reset()
    collector.episodes = [[] for _ in range(collector.env.num_envs)]
    collector.teacher_takeover.zero_()


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
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--resume", type=Path, default=None, help="resume all learner state from a run directory containing buffer/, sac/, and iql/")
    restore_group.add_argument("--restore-human-buffer", type=Path, default=None, help="load only successful human suffixes from a prior run, then retrain fresh IQL")
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
    run_dir = Path(args.logdir)
    writer = SummaryWriter(run_dir / "tb")
    generator = torch.Generator(device=gs.device).manual_seed(args.seed + 1)
    vector_step = 0
    try:
        if args.resume is not None:
            vector_step = load_run(args.resume, learner=learner, replay=replay, generator=generator)
        else:
            if args.restore_human_buffer is not None:
                restore_human_buffer(args.restore_human_buffer, replay)
            else:
                collect_initial_demos(collector, args.demo_episodes)
            learner.fit_normalizer_once()
            print(f"starting IQL pretraining for {config.iql_pretrain_updates} updates", flush=True)
            for update in range(config.iql_pretrain_updates):
                metrics = learner.pretrain_iql_update(replay.sample(config.batch_size, learner.device, human_suffix=True))
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
            save_networks(run_dir, learner=learner, vector_step=vector_step, generator=generator)
            save_buffer(run_dir, learner=learner, replay=replay, vector_step=vector_step)

        update_budget = 0.0
        for step in range(vector_step, args.steps):
            if len(replay) == replay.capacity:
                print(
                    f"replay buffer already full rows={len(replay)}/{replay.capacity} "
                    f"transitions={replay.total_transitions}; terminating cleanly",
                    flush=True,
                )
                break
            for episode_metric in collector.step(
                intervention_probability=args.teacher_intervention_prob,
                exploration_std=config.exploration_std,
                generator=generator,
            ):
                for key, value in episode_metric.items():
                    writer.add_scalar(key, value, step * args.num_envs)
            vector_step = step + 1
            if len(replay) == replay.capacity:
                save_networks(run_dir, learner=learner, vector_step=vector_step, generator=generator)
                save_buffer(run_dir, learner=learner, replay=replay, vector_step=vector_step)
                print(
                    f"replay buffer full rows={len(replay)}/{replay.capacity} "
                    f"transitions={replay.total_transitions}; terminating cleanly",
                    flush=True,
                )
                break
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
                if learner.update_count % 1_000 == 0:
                    save_networks(run_dir, learner=learner, vector_step=vector_step, generator=generator)
                update_budget -= 1.0
            if vector_step % 20_000 == 0:
                # A buffer snapshot always has matching network snapshots for
                # exact resume, even if the update cadence is changed later.
                save_networks(run_dir, learner=learner, vector_step=vector_step, generator=generator)
                save_buffer(run_dir, learner=learner, replay=replay, vector_step=vector_step)
            if step % 50 == 0:
                print(
                    f"train step={step} replay_rows={len(replay)}/{replay.capacity} "
                    f"transitions={replay.total_transitions} updates={learner.update_count}",
                    flush=True,
                )
        save_networks(run_dir, learner=learner, vector_step=vector_step, generator=generator)
        save_buffer(run_dir, learner=learner, replay=replay, vector_step=vector_step)
    except KeyboardInterrupt:
        save_networks(run_dir, learner=learner, vector_step=vector_step, generator=generator)
        save_buffer(run_dir, learner=learner, replay=replay, vector_step=vector_step)
        raise
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
