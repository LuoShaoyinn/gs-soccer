"""Training, deterministic full-clip evaluation, and explicitly synthetic smoke checks."""

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import JOINT_NAMES, ROOT, Config
from .geometry import from_rotation_vector
from .learner import Learner
from .motion import MotionLibrary, ResetSampler
from .replay import Replay
from .task import Task


def make_smoke(path):
    """Kinematic oscillations for exercising code, deliberately not walking demonstrations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timepoints = torch.arange(256) * 0.02
    phase = torch.sin(2 * torch.pi * 0.4 * timepoints)
    q = 0.015 * phase[:, None].expand(-1, len(JOINT_NAMES))
    pos = torch.zeros(256, 3)
    pos[:, 0] = 0.01 * phase
    pos[:, 2] = 0.56
    rot = torch.stack((0.02 * phase, 0.03 * phase, 0.1 * phase), -1)
    np.savez(
        path,
        robot="MOS9",
        fps=50.0,
        joint_names=np.array(JOINT_NAMES),
        joint_pos=q.numpy(),
        root_pos=pos.numpy(),
        root_quat=from_rotation_vector(rot).numpy(),
        smoke_only=True,
        provenance="Synthetic kinematic oscillation; NOT a walking reference",
    )
    return path


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != "mos9-supermimic-v1":
        raise ValueError("incompatible checkpoint format")
    return data


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def provenance():
    files = sorted((ROOT / "supermimic").glob("*.py"))
    code_hash = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in files)
    ).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {"source_sha256": code_hash, "git_commit": commit.stdout.strip()}


def save_checkpoint(
    path, cfg, library, learner, replay, sampler, iteration, budget, total_steps
):
    state = {
        "format": "mos9-supermimic-v1",
        "config": asdict(cfg),
        "motion_path": str(library.path),
        "fingerprint": library.fingerprint,
        "smoke_only": library.smoke_only,
        "learner": learner.state_dict(),
        "replay": replay.state_dict(),
        "sampler": sampler.state_dict(),
        "iteration": iteration,
        "budget": budget,
        "total_steps": total_steps,
        "provenance": provenance(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [
            np.random.get_state()[0],
            np.random.get_state()[1].tolist(),
            *np.random.get_state()[2:],
        ],
        "physics_resume": "episodes restart; replay boundary is cut",
    }
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def restore_rng(data):
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    random.setstate(data["python_rng"])
    name, keys, pos, gauss, cached = data["numpy_rng"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gauss, cached))


def init_genesis(cfg, backend, viewer):
    if not viewer:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    import genesis as gs

    if backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU requested but PyTorch cannot access CUDA/ROCm; use --backend cpu for debugging"
        )
    gs.init(
        backend=gs.gpu if backend == "gpu" else gs.cpu,
        seed=cfg.seed,
        logging_level="warning",
        performance_mode=True,
    )
    return torch.device(gs.device)


def train(args):
    if args.iterations is None:
        args.iterations = 2 if args.smoke_test else 10000
    cfg = Config(num_envs=args.num_envs, seed=args.seed)
    saved = load_checkpoint(args.resume) if args.resume else None
    initialization = None
    if saved:
        cfg = Config(**saved["config"])
    else:
        if args.robust_terrain:
            cfg.robustness_preset()
        if args.config:
            cfg = Config(**(asdict(cfg) | json.loads(args.config.read_text())))
        for name in (
            "batch_size",
            "imagination_batch",
            "replay_steps",
            "collect_steps",
            "model_batches",
        ):
            value = getattr(args, name)
            if value is not None:
                setattr(cfg, name, value)
        if args.paper_networks:
            cfg.paper_networks()
        if args.smoke_test:
            cfg.num_envs = 2
            cfg.model_widths, cfg.actor_widths, cfg.critic_widths = (
                (32, 32),
                (32, 32),
                (32,),
            )
            cfg.batch_size, cfg.imagination_batch, cfg.model_batches = 4, 4, 1
            cfg.critic_epochs, cfg.critic_minibatches = 1, 1
            cfg.replay_steps, cfg.collect_steps, cfg.update_ratio = 64, 16, 1 / 16
    cfg.validate()
    if args.iterations <= 0 or args.checkpoint_every <= 0:
        raise ValueError("iterations and checkpoint-every must be positive")
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError("run-dir must be new or empty; resume into a new directory")
    run.mkdir(parents=True, exist_ok=True)
    motion_path = args.motion or (Path(saved["motion_path"]) if saved else None)
    if args.smoke_test and not saved:
        motion_path = make_smoke(run / "smoke_reference.npz")
    if motion_path is None:
        raise ValueError(
            "--motion is required: supply MOS9-retargeted walking .npz clips (docs/supermimic.md)"
        )
    # Validate data before expensive simulator initialization.
    allow_smoke = args.smoke_test or (saved is not None and saved["smoke_only"])
    MotionLibrary(motion_path, Task(cfg), allow_smoke=allow_smoke)
    device = init_genesis(cfg, args.backend, args.viewer)
    seed_everything(cfg.seed)
    task = Task(cfg).to(device)
    library = MotionLibrary(motion_path, task, device, allow_smoke=allow_smoke)
    sampler, replay = ResetSampler(library, cfg.gamma), Replay(cfg.replay_steps)
    learner = Learner(task, library)
    if args.init_from:
        source = load_checkpoint(args.init_from)
        if (
            source["fingerprint"]["urdf"] != library.fingerprint["urdf"]
            or source["smoke_only"]
        ):
            raise ValueError(
                "warm start requires a real MOS9 tracker with the same URDF"
            )
        expanded = learner.warm_start(source["learner"])
        initialization = {
            "checkpoint": str(args.init_from.resolve()),
            "sha256": file_sha256(args.init_from),
            "parent_iteration": source["iteration"],
            "parent_transitions": source["total_steps"],
            "zero_extended_inputs": expanded,
            "optimizer_replay_sampler": "fresh",
        }
        del source
    from .environment import WalkingEnv

    env = WalkingEnv(task, library, sampler, args.viewer)
    iteration, budget, total_steps = 0, 0.0, 0
    if saved:
        if library.fingerprint != saved["fingerprint"]:
            raise ValueError(
                "reference motion or URDF checksum changed since checkpoint"
            )
        learner.load_state_dict(saved["learner"])
        replay.load_state_dict(saved["replay"])
        replay.cut_episodes()
        sampler.load_state_dict(saved["sampler"])
        iteration, budget, total_steps = (
            saved["iteration"],
            saved["budget"],
            saved["total_steps"],
        )
        restore_rng(saved)
        # Reset using restored sampling statistics. Simulator contacts and episodes start afresh.
        env.reset(torch.arange(cfg.num_envs, device=device))
        initialization = {
            "resume_checkpoint": str(args.resume.resolve()),
            "sha256": file_sha256(args.resume),
            "parent_iteration": iteration,
            "parent_transitions": total_steps,
            "optimizer_replay_sampler": "restored; physical episodes restart",
        }
        del saved
    metadata = {
        "config": asdict(cfg),
        "fingerprint": library.fingerprint,
        "smoke_only": library.smoke_only,
        "motion_path": str(library.path),
        "device": str(device),
        "provenance": provenance(),
        "hardware": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "method": "MOS9/Genesis adaptation of arXiv:2610.28480; not an official implementation",
        "initialization": initialization,
        "terrain": task.terrain.manifest() if cfg.robust else None,
    }
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if cfg.robust:
        np.save(run / "terrain_height_m.npy", task.terrain.grid.detach().cpu().numpy())
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run / "tensorboard"))
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    old_handlers = {
        s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    start = time.monotonic()
    stage_start_steps = total_steps
    print(
        f"{'SMOKE ONLY; not walking training' if library.smoke_only else 'MOS9 motion tracking'}: {device}, {cfg.num_envs} environments",
        flush=True,
    )
    save_checkpoint(
        run / "initial.pt",
        cfg,
        library,
        learner,
        replay,
        sampler,
        iteration,
        budget,
        total_steps,
    )
    try:
        for _ in range(args.iterations):
            metrics = {}
            for _ in range(cfg.collect_steps):
                reward, terminal, _truncated, terms = env.step(learner.actor, replay)
                total_steps += cfg.num_envs
                budget += cfg.update_ratio
                metrics.update(
                    {
                        "simulation/reward": float(reward.mean()),
                        "simulation/failure_rate": float(terminal.float().mean()),
                    }
                )
                metrics.update(
                    {"reward/" + k: float(v.mean()) for k, v in terms.items()}
                )
            while budget >= 1 and replay.size >= cfg.model_window:
                try:
                    batch = replay.sample(cfg.batch_size, cfg.model_window, device)
                except ValueError as exc:
                    metrics["model/windows_unavailable"] = 1.0
                    print(f"Waiting for within-episode windows: {exc}", flush=True)
                    budget = min(budget, 1.0)
                    break
                losses = []
                for model_batch in range(cfg.model_batches):
                    losses.append(learner.fit_model(batch))
                    if model_batch + 1 < cfg.model_batches:
                        batch = replay.sample(cfg.batch_size, cfg.model_window, device)
                metrics.update(
                    {k: sum(x[k] for x in losses) / len(losses) for k in losses[0]}
                )
                metrics.update(
                    learner.improve(replay.sample(cfg.imagination_batch, 1, device))
                )
                budget -= 1
            iteration += 1
            if env.completed:
                metrics["episodes/return"] = sum(
                    e["return"] for e in env.completed
                ) / len(env.completed)
                metrics["episodes/failure_rate"] = sum(
                    e["failed"] for e in env.completed
                ) / len(env.completed)
                with (run / "episodes.jsonl").open("a") as f:
                    for e in env.completed:
                        f.write(json.dumps({"iteration": iteration, **e}) + "\n")
                env.completed.clear()
            metrics.update(
                {
                    "training/updates": learner.updates,
                    "training/transitions": total_steps,
                    "training/wall_seconds": time.monotonic() - start,
                    "training/stage_transitions": total_steps - stage_start_steps,
                    "training/transitions_per_second": (total_steps - stage_start_steps)
                    / (time.monotonic() - start),
                }
            )
            metrics.update(env.diagnostics())
            if torch.cuda.is_available():
                metrics["resource/torch_peak_vram_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                )
            for key, value in metrics.items():
                writer.add_scalar(key, value, total_steps)
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"iteration": iteration, **metrics}) + "\n")
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "steps": total_steps,
                        "updates": learner.updates,
                        "reward": metrics["simulation/reward"],
                        "model_loss": metrics.get("model/state_loss"),
                    }
                ),
                flush=True,
            )
            if iteration % args.checkpoint_every == 0 or stop:
                save_checkpoint(
                    run / "latest.pt",
                    cfg,
                    library,
                    learner,
                    replay,
                    sampler,
                    iteration,
                    budget,
                    total_steps,
                )
            if stop:
                break
        save_checkpoint(
            run / "latest.pt",
            cfg,
            library,
            learner,
            replay,
            sampler,
            iteration,
            budget,
            total_steps,
        )
    finally:
        writer.close()
        env.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def evaluate(args):
    saved = load_checkpoint(args.checkpoint)
    cfg = Config(**saved["config"])
    cfg.num_envs, cfg.randomize, cfg.sensor_noise = 1, args.randomized, args.randomized
    cfg.push_enabled = args.pushes
    if args.seed is not None:
        cfg.seed = args.seed
    if args.flat:
        cfg.terrain_height = 0.0
    device = init_genesis(cfg, args.backend, args.viewer)
    seed_everything(cfg.seed)
    task = Task(cfg).to(device)
    path = args.motion or Path(saved["motion_path"])
    library = MotionLibrary(path, task, device, allow_smoke=saved["smoke_only"])
    if library.fingerprint["urdf"] != saved["fingerprint"]["urdf"]:
        raise ValueError("URDF changed since checkpoint")
    learner = Learner(task, library)
    learner.load_state_dict(saved["learner"])
    sampler = ResetSampler(library, cfg.gamma)
    from .environment import WalkingEnv

    env = WalkingEnv(task, library, sampler, args.viewer)
    rows = []
    try:
        for trial, clip in (
            (trial, clip)
            for trial in range(args.trials)
            for clip in range(len(library.names))
        ):
            name = library.names[clip]
            if not 0 <= args.start_frame < int(library.lengths[clip]) - 1:
                raise ValueError("start-frame must precede the final reference frame")
            env.reset(
                torch.tensor([0], device=device),
                torch.tensor([clip], device=device),
                torch.tensor([args.start_frame], device=device),
            )
            env.trajectories[0] = []
            initial_xy = env.state()[0, task.pos][:2].clone()
            errors, distance = [], []
            total = 0.0
            duration_frames = int(library.lengths[clip]) - 1 - args.start_frame
            for frame in range(duration_frames):
                reward, terminal, _truncated, _ = env.step(
                    learner.actor, deterministic=True, auto_reset=False
                )
                state = env.state()
                reference = library.reference(env.clip, env.frame, context=env.context)[
                    :, 0
                ]
                errors.append(
                    float((state[:, task.pos] - reference[:, task.pos]).norm())
                )
                distance.append(state[0, task.pos][:2].tolist())
                total += float(reward[0])
                if terminal[0]:
                    break
            displacement = [distance[-1][i] - float(initial_xy[i]) for i in range(2)]
            rows.append(
                {
                    "clip": name,
                    "trial": trial,
                    "start_frame": args.start_frame,
                    "frames": frame + 1,
                    "failed": bool(terminal[0]),
                    "return": total,
                    "normalized_return": total / (5 * cfg.dt * duration_frames),
                    "root_position_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "displacement_xy": displacement,
                }
            )
    finally:
        env.close()
    result = {
        "smoke_only": library.smoke_only,
        "training_library": library.fingerprint == saved["fingerprint"],
        "failures": sum(r["failed"] for r in rows),
        "clips": rows,
        "fingerprint": library.fingerprint,
        "checkpoint": str(args.checkpoint.resolve()),
        "settings": {
            "randomized": cfg.randomize,
            "sensor_noise": cfg.sensor_noise,
            "pushes": cfg.push_enabled,
            "seed": cfg.seed,
            "terrain": task.terrain.manifest() if cfg.robust else None,
        },
        "pushes_total": env.pushes_total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="SuperMimic MOS9 walking experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "validate", help="validate MOS9 reference data without starting Genesis"
    )
    p.add_argument("--motion", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--motion", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--init-from",
        type=Path,
        help="warm start weights; fresh optimizer/replay; compatible privileged inputs may grow",
    )
    p.add_argument(
        "--robust-terrain",
        action="store_true",
        help="5 cm uniform terrain, larger dynamics variation and random chunk impulses",
    )
    p.add_argument(
        "--config", type=Path, help="JSON configuration overrides, recorded in metadata"
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="synthetic pipeline test; never evidence of walking",
    )
    p.add_argument("--iterations", type=int, help="default: 10000; smoke-test: 2")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--paper-networks", action="store_true")
    for name in (
        "batch-size",
        "imagination-batch",
        "replay-steps",
        "collect-steps",
        "model-batches",
    ):
        p.add_argument("--" + name, type=int)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--viewer", action="store_true")
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--viewer", action="store_true")
    p.add_argument(
        "--flat",
        action="store_true",
        help="flat-ground control evaluation for a terrain checkpoint",
    )
    p.add_argument(
        "--randomized",
        action="store_true",
        help="sample the training dynamics and sensor noise",
    )
    p.add_argument(
        "--pushes",
        action="store_true",
        help="apply saved random chunk impulse settings",
    )
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--seed",
        type=int,
        help="also regenerates a fresh terrain for terrain checkpoints",
    )
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.resume and (args.init_from or args.robust_terrain or args.config):
                raise ValueError(
                    "resume restores saved configuration; use init-from for a new distribution"
                )
            train(args)
        elif args.command == "eval":
            if args.trials < 1:
                raise ValueError("trials must be positive")
            evaluate(args)
        else:
            cfg = Config()
            library = MotionLibrary(args.motion, Task(cfg))
            print(
                json.dumps(
                    {
                        "clips": library.names,
                        "frames": library.lengths.tolist(),
                        "fingerprint": library.fingerprint,
                    },
                    indent=2,
                )
            )
    except (ValueError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")
