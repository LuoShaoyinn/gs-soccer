import tempfile
import unittest
from pathlib import Path

import torch

from supermimic.config import Config
from supermimic.learner import Learner
from supermimic.motion import MotionLibrary
from supermimic.replay import Replay
from supermimic.run import make_smoke
from supermimic.task import Task
from supermimic.terrain import UniformTerrain


class RobustContracts(unittest.TestCase):
    def test_uniform_distribution_bounds_reproducibility_and_height_jacobian(self):
        cfg = Config(robust=True, terrain_height=0.05)
        terrain = UniformTerrain(cfg)
        torch.testing.assert_close(terrain.grid, UniformTerrain(cfg).grid)
        self.assertGreater(float(terrain.grid.max()), 0.049)
        self.assertLess(float(terrain.grid.min()), -0.049)
        self.assertTrue((terrain.grid.abs() <= 0.05).all())
        xy = torch.tensor([[0.013, 0.021], [-1.123, 1.236]], requires_grad=True)
        h = terrain.height(xy).sum()
        gradient = torch.autograd.grad(h, xy)[0]
        direction = torch.tensor([[0.3, -0.1], [0.2, 0.1]])
        eps = 1e-4
        finite = (
            terrain.height(xy + eps * direction).sum()
            - terrain.height(xy - eps * direction).sum()
        ) / (2 * eps)
        torch.testing.assert_close(
            (gradient * direction).sum(), finite, atol=1e-3, rtol=0.02
        )

    def test_warm_start_preserves_predictions_and_shifted_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_smoke(Path(tmp) / "motion.npz")
            old_cfg = Config(
                model_widths=(32, 32), actor_widths=(32,), critic_widths=(32,)
            )
            old_task = Task(old_cfg)
            old_lib = MotionLibrary(path, old_task, allow_smoke=True)
            old = Learner(old_task, old_lib)
            cfg = Config(**vars(old_cfg))
            cfg.robust, cfg.terrain_height = True, 0.05
            task = Task(cfg)
            lib = MotionLibrary(path, task, allow_smoke=True)
            new = Learner(task, lib)
            expanded = new.warm_start(old.state_dict())
            self.assertEqual(len(expanded), 2)
            s = old_lib.states[0, :3]
            c = torch.zeros(3, old_task.context_dim)
            larger = torch.zeros(3, task.context_dim)
            larger[:, : old_task.context_dim] = c
            a = torch.randn(3, task.n)
            torch.testing.assert_close(old.model(s, a, c), new.model(s, a, larger))
            refs = lib.reference(torch.zeros(3, dtype=torch.long), torch.arange(3), 10)
            goal = task.goal(s, refs)
            history = task.observation(s)[:, None].expand(-1, cfg.history, -1)
            torch.testing.assert_close(
                old.actor(history, goal), new.actor(history, goal)
            )
            torch.testing.assert_close(
                old.critic(s, goal, c), new.critic(s, goal, larger)
            )
            larger[:, task.offset_context] = torch.tensor([1.0, -2.0, 0.03])
            shifted = lib.reference(
                torch.zeros(3, dtype=torch.long), torch.arange(3), 10, context=larger
            )
            torch.testing.assert_close(
                shifted[..., task.pos],
                refs[..., task.pos] + torch.tensor([1.0, -2.0, 0.03]),
            )
            torch.testing.assert_close(shifted[..., task.vel], refs[..., task.vel])

    def test_impulse_cut_is_per_environment_without_false_failure(self):
        replay = Replay(12)
        for i in range(8):
            replay.append(
                action=torch.zeros(2, 18),
                done=torch.zeros(2, dtype=torch.bool),
                terminal=torch.zeros(2, dtype=torch.bool),
                frame=torch.full((2,), i),
            )
        replay.cut_episodes(torch.tensor([1]))
        self.assertFalse(replay.data["done"][7, 0])
        self.assertTrue(replay.data["done"][7, 1])
        self.assertFalse(replay.data["terminal"][: replay.size].any())


if __name__ == "__main__":
    unittest.main()
