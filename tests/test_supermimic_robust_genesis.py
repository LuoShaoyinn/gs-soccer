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
    os.environ.get("GS_SUPERMIMIC_GPU_TEST") == "1", "opt-in GPU physics"
)
class RobustGenesisContracts(unittest.TestCase):
    def test_terrain_gains_partial_reset_and_world_velocity_impulse(self):
        cfg = Config(num_envs=4)
        cfg.robustness_preset()
        cfg.sensor_noise = False
        device = init_genesis(cfg, "gpu", False)
        task = Task(cfg).to(device)
        from supermimic.environment import WalkingEnv

        with tempfile.TemporaryDirectory() as tmp:
            library = MotionLibrary(
                make_smoke(Path(tmp) / "motion.npz"), task, device, allow_smoke=True
            )
            env = WalkingEnv(task, library, ResetSampler(library, cfg.gamma))
            try:
                idx = torch.arange(4, device=device)
                kp = env.robot.get_dofs_kp(env.indices).clone()
                kd = env.robot.get_dofs_kv(env.indices).clone()
                torch.testing.assert_close(
                    kp, cfg.kp * (1 + env.context[:, task.kp_context])
                )
                torch.testing.assert_close(
                    kd, cfg.kd * (1 + env.context[:, task.kd_context])
                )
                xy = torch.rand(4, 25, 2, device=device) * 4 - 2
                physics = env.floor.get_terrain_height(xy)
                torch.testing.assert_close(
                    physics, task.terrain.height(xy), atol=2e-6, rtol=1e-4
                )
                context = env.context.clone()
                state = env.state().clone()
                env.reset(
                    idx[:1],
                    torch.tensor([0], device=device),
                    torch.tensor([30], device=device),
                )
                torch.testing.assert_close(
                    env.robot.get_dofs_kp(env.indices)[1:], kp[1:]
                )
                torch.testing.assert_close(
                    env.robot.get_dofs_kv(env.indices)[1:], kd[1:]
                )
                torch.testing.assert_close(env.context[1:], context[1:])
                torch.testing.assert_close(env.state()[1:], state[1:])
                replay = Replay(16)
                replay.append(
                    done=torch.zeros(4, dtype=torch.bool),
                    terminal=torch.zeros(4, dtype=torch.bool),
                )
                before = env.state().clone()
                delta = torch.tensor([[0.4, -0.2, 0.3]], device=device)
                env.apply_push(idx[1:2], replay, delta)
                after = env.state()
                torch.testing.assert_close(after[:, task.pos], before[:, task.pos])
                torch.testing.assert_close(
                    after[1, task.vel][:2] - before[1, task.vel][:2],
                    delta[0, :2],
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    after[1, task.ang] - before[1, task.ang],
                    torch.tensor([0.0, 0.0, 0.3], device=device),
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(after[[0, 2, 3]], before[[0, 2, 3]])
                self.assertTrue(replay.data["done"][0, 1])
                self.assertFalse(replay.data["terminal"][: replay.size].any())

                class ZeroActor:
                    def __call__(self, history, goal):
                        return torch.zeros(len(history), task.n, device=device)

                for _ in range(10):
                    env.step(ZeroActor(), deterministic=True)
                self.assertTrue(torch.isfinite(env.state()).all())
                print(
                    "Verified terrain triangle heights, per-environment gains, isolated reset and exact world xy/yaw impulse",
                    flush=True,
                )
            finally:
                env.close()
