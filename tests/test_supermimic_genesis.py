"""Opt-in real Genesis test: GS_SUPERMIMIC_GPU_TEST=1 uv run --extra rocm python -m unittest discover -s tests -v."""

import os
import tempfile
import unittest
from pathlib import Path

import torch

from supermimic.config import Config
from supermimic.motion import MotionLibrary, ResetSampler
from supermimic.replay import Replay
from supermimic.run import init_genesis, make_smoke
from supermimic.task import Task


@unittest.skipUnless(
    os.environ.get("GS_SUPERMIMIC_GPU_TEST") == "1",
    "set GS_SUPERMIMIC_GPU_TEST=1 for actual GPU physics",
)
class GenesisContracts(unittest.TestCase):
    def test_reset_velocities_fk_partial_reset_and_terminal_transition(self):
        from supermimic.environment import WalkingEnv

        cfg = Config(num_envs=2, sensor_noise=False, randomize=False)
        device = init_genesis(cfg, "gpu", False)
        task = Task(cfg).to(device)
        with tempfile.TemporaryDirectory() as tmp:
            library = MotionLibrary(
                make_smoke(Path(tmp) / "smoke.npz"), task, device, allow_smoke=True
            )
            sampler = ResetSampler(library, cfg.gamma)
            env = WalkingEnv(task, library, sampler)
            try:
                idx = torch.arange(2, device=device)
                clips = torch.zeros(2, dtype=torch.long, device=device)
                frames = torch.tensor([12, 35], device=device)
                env.reset(idx, clips, frames)
                reference = library.reference(clips, frames)[:, 0]
                torch.testing.assert_close(env.state(), reference, atol=1e-5, rtol=1e-4)
                errors = env.validate_kinematics()
                self.assertLess(max(errors), 1e-3)
                env.previous[1] = 0.123
                before, history = env.state().clone(), env.history.clone()
                env.reset(idx[:1], clips[:1], frames[:1])
                torch.testing.assert_close(env.state()[1], before[1])
                torch.testing.assert_close(env.history[1], history[1])

                class ConstantActor:
                    def __call__(self, history, goal):
                        return torch.full((len(history), task.n), 10.0, device=device)

                replay = Replay(16)
                cfg.height_threshold = 1e-8
                _, terminal, _, _ = env.step(
                    ConstantActor(), replay, deterministic=True
                )
                self.assertTrue(terminal.all())
                torch.testing.assert_close(
                    replay.data["next_state"][0, :, task.previous],
                    replay.data["action"][0],
                )
                self.assertTrue((replay.data["action"][0] == 10).all())
                self.assertFalse(
                    torch.allclose(replay.data["next_state"][0], env.state().cpu())
                )
                self.assertTrue(replay.data["done"][0].all())
                self.assertTrue((env.steps == 0).all())
                print(
                    f"Genesis authored-frame FK errors (position/rotation/velocity/angular): {errors}"
                )
            finally:
                env.close()
