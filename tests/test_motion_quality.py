import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from motion_tools.quality import RobotModel, lowest_point


class GeometryTests(unittest.TestCase):
    def test_support_heights_for_rotated_cylinders(self):
        r = Rotation.from_euler(
            "xyz", [[0, 0, 0], [0, np.pi / 2, 0], [0.3, 0.7, 0.4]]
        ).as_matrix()
        p = np.zeros((3, 3))
        result = lowest_point(p, r, "cylinder", [0.02, 0.16])
        c = np.abs(r[:, 2, 2])
        expected = -0.08 * c - 0.02 * np.sqrt(1 - c * c)
        np.testing.assert_allclose(result[:, 2], expected, atol=1e-12)

    def test_mesh_skipping_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "primitive"):
            RobotModel("assets/g1/g1_29dof.urdf")
        m = RobotModel("assets/g1/g1_29dof.urdf", skip_meshes=True)
        self.assertEqual(len(m.names), 29)

    def test_numpy_fk_matches_independently_tested_torch_fk(self):
        import torch

        from supermimic.geometry import Kinematics

        path = "assets/MOS9/MOS9_walk.urdf"
        model = RobotModel(path)
        rng = np.random.default_rng(3)
        q = rng.uniform(-0.1, 0.1, (5, 18))
        p = rng.normal(size=(5, 3))
        r = Rotation.random(5, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
        expected = model.fk(q, p, r)
        fk = Kinematics(path, model.names, model.links).double()
        actual = fk(
            torch.tensor(q),
            torch.zeros(5, 18).double(),
            torch.tensor(p),
            torch.tensor(r),
            torch.zeros(5, 3).double(),
            torch.zeros(5, 3).double(),
        )[0]
        np.testing.assert_allclose(
            actual.numpy(),
            np.stack([expected[n][0] for n in model.links], axis=1),
            atol=2e-7,
        )


if __name__ == "__main__":
    unittest.main()
