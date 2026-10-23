import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from motion_tools.quality import RobotModel
from motion_tools.retarget import TorchModel, fit, save, sole_offset
from supermimic.config import Config
from supermimic.motion import MotionLibrary
from supermimic.task import Task


class RetargetTests(unittest.TestCase):
    def test_torch_fk_and_joint_jacobian_match_finite_difference(self):
        m = RobotModel("assets/MOS9/MOS9_walk.urdf")
        tm = TorchModel(m)
        q = torch.full((2, 18), 0.02, dtype=torch.float64, requires_grad=True)
        p = torch.zeros(2, 3, dtype=torch.float64)
        r = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
        poses = tm.fk(q, p, r)
        expected = m.fk(q.detach().numpy(), p.numpy(), np.array([[1.0, 0, 0, 0]] * 2))
        np.testing.assert_allclose(
            poses["Rfoot"][0].detach(), expected["Rfoot"][0], atol=1e-12
        )
        objective = poses["Rfoot"][0].square().sum()
        grad = torch.autograd.grad(objective, q)[0]
        eps = 1e-5
        delta = torch.zeros_like(q)
        delta[:, m.names.index("right_knee")] = eps
        plus = tm.fk(q + delta, p, r)["Rfoot"][0].square().sum()
        minus = tm.fk(q - delta, p, r)["Rfoot"][0].square().sum()
        torch.testing.assert_close(
            grad[:, m.names.index("right_knee")].sum(), (plus - minus) / (2 * eps)
        )

    def test_stance_fit_and_export_are_valid_for_training_loader(self):
        m = RobotModel("assets/MOS9/MOS9_walk.urdf")
        t = 64
        offsets, target_rot, points = zip(
            *(sole_offset(m, n) for n in ("Rfoot", "Lfoot"))
        )
        root = np.zeros((t, 3))
        root[:, 2] = 0.53
        feet = np.repeat(np.array(points)[None], t, axis=0) + root[:, None]
        reference = {
            "root_pos": root,
            "root_rotation": np.repeat(np.eye(3)[None], t, axis=0),
            "feet": feet,
            "foot_rotation": np.repeat(np.array(target_rot)[None], t, axis=0),
            "contacts": np.ones((t, 2)),
            "sole_offsets": offsets,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            q, p, metrics = fit(m, reference, iterations=150)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue((q >= m.limits[:, 0]).all())
        self.assertTrue((q <= m.limits[:, 1]).all())
        self.assertLess(metrics["foot_position_error_p95_m"], 0.02)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stance.npz"
            r = np.repeat(np.array([[1.0, 0, 0, 0]]), t, axis=0)
            save(path, m, q, p, r, {"test": True})
            lib = MotionLibrary(path, Task(Config()))
            self.assertEqual(int(lib.lengths[0]), t)
            with np.load(path) as data:
                self.assertEqual(
                    str(data["quality_status"]),
                    "kinematic_candidate_not_dynamic_validation",
                )


if __name__ == "__main__":
    unittest.main()
