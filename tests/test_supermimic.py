import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic.config import Config
from supermimic.geometry import from_rotation_vector, rotate, rotation_error_squared
from supermimic.learner import Learner, lambda_returns
from supermimic.motion import MotionLibrary, ResetSampler
from supermimic.networks import WorldModel, frozen
from supermimic.replay import Replay
from supermimic.run import load_checkpoint, make_smoke, restore_rng, save_checkpoint
from supermimic.task import Task


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = Config(
            num_envs=2,
            model_widths=(32, 32),
            actor_widths=(32, 32),
            critic_widths=(32,),
            batch_size=4,
            imagination_batch=4,
            critic_epochs=1,
            critic_minibatches=1,
        )
        self.task = Task(self.cfg)
        self.motion_path = make_smoke(self.path / "reference.npz")
        self.library = MotionLibrary(self.motion_path, self.task, allow_smoke=True)

    def rewrite(self, **changes):
        with np.load(self.motion_path) as d:
            data = dict(d)
        data.update(changes)
        np.savez(self.motion_path, **data)

    def test_motion_rejects_wrong_robot_smoke_nonfinite_and_bad_rotation(self):
        with self.assertRaisesRegex(ValueError, "smoke-only"):
            MotionLibrary(self.motion_path, self.task)
        self.rewrite(robot="G1")
        with self.assertRaisesRegex(ValueError, "retarget"):
            MotionLibrary(self.motion_path, self.task, allow_smoke=True)
        make_smoke(self.motion_path)
        self.rewrite(root_pos=np.full((256, 3), np.nan))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            MotionLibrary(self.motion_path, self.task, allow_smoke=True)
        make_smoke(self.motion_path)
        self.rewrite(root_quat=np.zeros((256, 4)))
        with self.assertRaisesRegex(ValueError, "unit wxyz"):
            MotionLibrary(self.motion_path, self.task, allow_smoke=True)

    def test_named_joint_reordering_and_quaternion_signs(self):
        with np.load(self.motion_path) as d:
            data = dict(d)
        data["joint_names"] = data["joint_names"][::-1]
        data["joint_pos"] = data["joint_pos"][:, ::-1]
        data["root_quat"][::2] *= -1
        np.savez(self.motion_path, **data)
        loaded = MotionLibrary(self.motion_path, self.task, allow_smoke=True)
        torch.testing.assert_close(loaded.states, self.library.states)

    def test_loader_rejects_units_limits_rate_and_short_data(self):
        for key, value, message in (
            ("fps", 60.0, "resample"),
            ("joint_pos", np.ones((256, 18)) * 10, "limits"),
            ("joint_pos", np.zeros((8, 18)), "too short"),
        ):
            make_smoke(self.motion_path)
            self.rewrite(**{key: value})
            with self.assertRaisesRegex(ValueError, message):
                MotionLibrary(self.motion_path, self.task, allow_smoke=True)

    def test_reference_future_never_crosses_clip(self):
        clips, frames = torch.tensor([0, 0]), torch.tensor([0, 255])
        refs = self.library.reference(clips, frames, 10)
        torch.testing.assert_close(refs[1], self.library.states[0, -1].expand(10, -1))

    def test_tracking_reward_and_physical_termination(self):
        t = self.task
        state = self.library.states[0, :2].clone()
        zero = torch.zeros(2, t.n)
        reward, _ = t.reward(state, zero, zero, state)
        torch.testing.assert_close(reward, torch.full((2,), 0.1))
        fallen = state.clone()
        fallen[0, t.pos.start + 2] -= 0.2
        fallen[1, t.quat] = from_rotation_vector(torch.tensor([1.5, 0.0, 0.0]))
        self.assertTrue(t.terminal(fallen, state).all())
        self.assertFalse(t.terminal(state, state).any())

    def test_quaternions_and_fk_have_finite_identity_gradients(self):
        q = torch.zeros(3, 3, requires_grad=True)
        rot = from_rotation_vector(q)
        loss = (
            rotation_error_squared(rot, rot.detach()).sum()
            + rotate(rot, torch.ones(3, 3)).sum()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        state = self.library.states[0, :2].clone().requires_grad_(True)
        self.task.bodies(state)[0].square().sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad[:, self.task.q].abs().sum()), 0.0)

    def test_body_velocity_matches_finite_difference(self):
        t = self.task
        state = self.library.states[0, 3:4].clone()
        p, _r, v, _w = t.bodies(state)
        eps = 1e-4
        next_state = state.clone()
        next_state[:, t.q] += state[:, t.dq] * eps
        next_state[:, t.pos] += state[:, t.vel] * eps
        from supermimic.geometry import multiply

        next_state[:, t.quat] = multiply(
            from_rotation_vector(state[:, t.ang] * eps), state[:, t.quat]
        )
        finite_difference = (t.bodies(next_state)[0] - p) / eps
        torch.testing.assert_close(finite_difference, v, atol=1.5e-3, rtol=0.02)

    def test_world_model_jacobian_matches_finite_difference(self):
        t = self.task
        model = WorldModel(t)
        state = self.library.states[0, :1]
        context = torch.zeros(1, t.context_dim)
        action = torch.randn(1, t.n, requires_grad=True) * 0.1
        direction = torch.randn_like(action)
        with frozen(model):
            objective = t.bodies(model(state, action, context))[0].square().sum()
            gradient = torch.autograd.grad(objective, action)[0]
            epsilon = 0.01
            plus = (
                t.bodies(model(state, action + epsilon * direction, context))[0]
                .square()
                .sum()
            )
            minus = (
                t.bodies(model(state, action - epsilon * direction, context))[0]
                .square()
                .sum()
            )
            torch.testing.assert_close(
                (gradient * direction).sum(),
                (plus - minus) / (2 * epsilon),
                atol=2e-5,
                rtol=0.05,
            )
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_model_is_xy_translation_equivariant_and_logs_action(self):
        t = self.task
        model = WorldModel(t)
        s = self.library.states[0, :2]
        shifted = s.clone()
        shifted[:, t.pos.start : t.pos.start + 2] += torch.tensor([20.0, -10.0])
        a, c = torch.randn(2, t.n), torch.zeros(2, t.context_dim)
        original = model(s, a, c)
        translated = model(shifted, a, c)
        translated[:, t.pos.start : t.pos.start + 2] -= torch.tensor([20.0, -10.0])
        torch.testing.assert_close(original, translated, atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(original[:, t.previous], a)

    def test_gae_bootstrap_and_death(self):
        reward = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        death = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        values = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        result = lambda_returns(reward, death, values, 0.9, 1.0)
        torch.testing.assert_close(result, torch.tensor([[10.9, 1.0], [11.0, 11.0]]))

    def replay_row(self, frame, done=False):
        n = 2
        return {
            "action": torch.full((n, 18), float(frame)),
            "done": torch.full((n,), done),
            "frame": torch.full((n,), frame),
            "state": torch.full((n, 67), float(frame)),
        }

    def test_replay_no_reset_crossing_or_ring_chronology_loss(self):
        replay = Replay(12)
        for i in range(30):
            replay.append(**self.replay_row(i, i % 5 == 4))
        b = replay.sample(200, 4, "cpu")
        self.assertTrue((b["frame"] >= 18).all())
        self.assertTrue((b["frame"][:, 1:] - b["frame"][:, :-1] == 1).all())
        self.assertFalse(b["done"][:, :-1].any())
        copy = Replay(12)
        copy.load_state_dict(replay.state_dict())
        copy.cut_episodes()
        self.assertTrue(copy.data["done"][(copy.cursor - 1) % 12].all())
        with self.assertRaisesRegex(ValueError, "within-episode"):
            r = Replay(4)
            for i in range(4):
                r.append(**self.replay_row(i, True))
            r.sample(4, 2, "cpu")

    def test_full_32_step_update_freezes_world_model(self):
        t = self.task
        learner = Learner(t, self.library)
        states = self.library.states[0, :4]
        batch = {
            "state": states[:, None],
            "context": torch.zeros(4, 1, t.context_dim),
            "clip": torch.zeros(4, 1, dtype=torch.long),
            "frame": torch.arange(4)[:, None],
            "history": t.observation(states)[:, None, None].expand(
                -1, 1, self.cfg.history, -1
            ),
        }
        model_before = {k: p.clone() for k, p in learner.model.named_parameters()}
        actor_before = {k: p.clone() for k, p in learner.actor.named_parameters()}
        logs = learner.improve(batch)
        self.assertTrue(all(np.isfinite(v) for v in logs.values()))
        self.assertGreater(logs["policy/gradient_norm"], 0)
        self.assertTrue(
            all(
                torch.equal(p, model_before[k])
                for k, p in learner.model.named_parameters()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(p, actor_before[k])
                for k, p in learner.actor.named_parameters()
            )
        )
        self.assertTrue(all(p.grad is None for p in learner.model.parameters()))

    def test_autoregressive_model_fit_and_checkpoint_rng_roundtrip(self):
        t = self.task
        learner = Learner(t, self.library)
        states = self.library.states[0, :9]
        batch = {
            "state": states[:-1][None].expand(2, -1, -1),
            "next_state": states[1:][None].expand(2, -1, -1),
            "action": torch.zeros(2, 8, t.n),
            "context": torch.zeros(2, 8, t.context_dim),
            "terminal": torch.zeros(2, 8, dtype=torch.bool),
            "clip": torch.zeros(2, 8, dtype=torch.long),
            "frame": torch.arange(8)[None].expand(2, -1),
        }
        logs = learner.fit_model(batch)
        self.assertTrue(all(np.isfinite(v) for v in logs.values()))
        replay, sampler = Replay(16), ResetSampler(self.library, self.cfg.gamma)
        replay.append(**self.replay_row(0))
        path = self.path / "latest.pt"
        save_checkpoint(
            path, self.cfg, self.library, learner, replay, sampler, 3, 0.5, 10
        )
        expected = torch.rand(3)
        saved = load_checkpoint(path)
        restore_rng(saved)
        torch.testing.assert_close(torch.rand(3), expected)
        restored = Learner(t, self.library)
        restored.load_state_dict(saved["learner"])
        for a, b in zip(learner.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(a, b)

    def test_return_sampler_learns_bins_and_samples_valid_frames(self):
        sampler = ResetSampler(self.library, 0.99)
        sampler.update([(0, f, 0.05) for f in range(100)])
        self.assertEqual(int(sampler.seen.sum()), 2)
        self.assertTrue((sampler.returns[sampler.seen] > 0).all())
        clips, frames = sampler.sample(1000)
        self.assertTrue((frames < self.library.lengths[clips] - 1).all())


if __name__ == "__main__":
    unittest.main()
