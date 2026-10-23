import tempfile, unittest
from pathlib import Path
import torch
from algorithm.mos9.task import TaskState, observation
from algorithm.mos9.replay import Replay
from algorithm.mos9.learner import Actor, Critic, Settings, Learner, bellman
from algorithm.mos9.data import episodes, learner_transition


def rows(n, terminal=True):
    obs = torch.zeros(n, 63)
    obs[:, 6:9] = 0.4
    nextobs = obs.clone()
    return dict(
        observation=obs,
        next_observation=nextobs,
        action=torch.zeros(n, 18),
        teacher_control=torch.ones(n, dtype=torch.bool),
        reward=torch.linspace(0.001, 0.015, n),
        terminal=torch.arange(n) == n - 1
        if terminal
        else torch.zeros(n, dtype=torch.bool),
        success=torch.arange(n) == n - 1
        if terminal
        else torch.zeros(n, dtype=torch.bool),
    )


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.state = TaskState(1, "cpu", torch.zeros(18), 0.5596)
        self.state.reset(torch.tensor([0]))
        self.state.command[:] = 0.4
        self.kw = dict(
            pos=torch.tensor([[0.0, 0.0, 0.5396]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            velocity=torch.tensor([[0.4, 0.4, 0.0]]),
            angular=torch.tensor([[0.0, 0.0, 0.4]]),
            torque=torch.zeros(1, 18),
            torque_limits=torch.ones(18),
        )

    def test_full_filter_and_consecutive_hold(self):
        for _ in range(348):
            r, done, success, _, _ = self.state.advance(**self.kw)
            self.assertFalse(done.item())
        r, done, success, _, out = self.state.advance(**self.kw)
        self.assertTrue(success.item())
        self.assertEqual(out[0]["steps"], 349)
        self.assertEqual(out[0]["best_hold_seconds"], 5)
        self.assertAlmostEqual(r.item(), 0.02, places=6)

    def test_standing_cannot_succeed(self):
        self.kw["velocity"].zero_()
        self.kw["angular"].zero_()
        for _ in range(1000):
            r, done, success, _, out = self.state.advance(**self.kw)
        self.assertTrue(done.item())
        self.assertFalse(success.item())
        self.assertTrue(out[0]["timed_out"])

    def test_fall_overrides_hold_and_deadline(self):
        self.state.steps[:] = 999
        self.state.hold[:] = 249
        self.state.history[:, :, 0:2] = 0.4
        self.state.history[:, :, 3] = 0.4
        self.kw["pos"][:, 2] = 0.1
        r, done, success, terms, out = self.state.advance(**self.kw)
        self.assertEqual(r.item(), -5)
        self.assertFalse(success.item())
        self.assertTrue(out[0]["fallen"])

    def test_reward_bounds(self):
        self.kw["velocity"][:] = 100
        self.kw["angular"][:] = 100
        self.kw["torque"][:] = 100
        r, *_ = self.state.advance(**self.kw)
        self.assertGreaterEqual(r.item(), -0.010001)
        self.assertLessEqual(r.item(), 0.02)

    def test_observation_world_to_body_and_layout(self):
        q = torch.ones(1, 18)
        dq = torch.ones_like(q) * 2
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        o = observation(
            q,
            dq,
            quat,
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.zeros(18),
            torch.ones_like(q) * 4,
            torch.ones(1, 3) * 0.5,
        )
        self.assertEqual(o.shape, (1, 63))
        torch.testing.assert_close(o[0, :3], torch.tensor([0.2, 0.4, 0.6]))
        torch.testing.assert_close(o[0, 3:6], torch.tensor([0.0, 0.0, -1.0]))
        self.assertTrue(torch.all(o[:, 45:] == 4))


class ReplayTests(unittest.TestCase):
    def make(self, path, **kw):
        return Replay(
            path,
            capacity=512,
            block_size=8,
            ram_blocks=64,
            batch_size=32,
            device="cpu",
            gpu_cache_bytes=0,
            **kw,
        )

    def test_unfinished_not_reference_and_all_heads_factual(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(p)
            d = rows(24)
            ids = r.append(d)
            r.store.flush()
            self.assertFalse(r.labels["eligible"].any())
            r.confirm(ids, d["reward"])
            calls = 0
            original = r.store.sample_async

            def counted():
                nonlocal calls
                calls += 1
                return original()

            r.store.sample_async = counted
            b = r.sample(True)
            self.assertGreater(calls, 0)
            self.assertEqual(b["observation"].dtype, torch.float32)
            self.assertEqual(r.store._schema["observation"][1], torch.float16)
            for i, rowid in enumerate(b["row_id"].tolist()):
                for h in (1, 2, 10, 24, 1000):
                    expected = sum(
                        float(d["reward"][j]) * 0.99 ** (j - rowid)
                        for j in range(rowid, min(rowid + h, 24))
                    )
                    self.assertAlmostEqual(
                        float(b["reference_target"][i, h - 1]), expected, places=6
                    )
            r.close()

    def test_interleaved_success_suffix(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(p)
            d = rows(20)
            d["terminal"].zero_()
            d["success"].zero_()
            d["terminal"][18] = True
            d["success"][18] = True
            ids = r.append(d)
            r.confirm(ids[4::2], d["reward"][4::2])
            b = r.sample(True)
            self.assertTrue(torch.all(b["row_id"] % 2 == 0))
            self.assertTrue(torch.all(b["row_id"] >= 4))
            for i, j in enumerate(b["row_id"].tolist()):
                self.assertAlmostEqual(
                    float(b["reference_target"][i, 0]), float(d["reward"][j]), places=6
                )
            r.close()

    def test_checkpoint_prefix_and_labels(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(Path(p) / "source")
            d = rows(24)
            ids = r.append(d)
            r.confirm(ids, d["reward"])
            state = r.state_dict()
            r.append(rows(8))
            r.store.flush()
            other = self.make(Path(p) / "restore")
            other.restore(state)
            self.assertEqual(other.size, 24)
            self.assertTrue(other.labels["eligible"][:24].all())
            self.assertFalse(other.labels["eligible"][24:].any())
            b = other.sample(True)
            self.assertTrue(torch.all(b["row_id"] < 24))
            r.close()
            other.close()

    def test_frequent_flushes_keep_all_accepted_rows(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(p)
            for _ in range(100):
                r.append(rows(1))
                r.store.flush()
            self.assertEqual(len(r.store), 100)
            self.assertEqual(len(r.store._blocks), 100)
            r.close()

    def test_failed_or_student_suffix_cannot_be_confirmed(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(p)
            d = rows(24)
            d["success"].zero_()
            ids = r.append(d)
            with self.assertRaises(ValueError):
                r.confirm(ids, d["reward"])
            d = rows(24)
            d["teacher_control"].zero_()
            ids = r.append(d)
            with self.assertRaises(ValueError):
                r.confirm(ids, d["reward"])
            self.assertFalse(r.labels["eligible"].any())
            r.close()

    def test_reference_holdout_is_disjoint_and_restored(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(Path(p) / "source")
            d = rows(24)
            d["reference_validation"] = torch.ones(24, dtype=torch.bool)
            ids = r.append(d)
            r.confirm(ids, d["reward"])
            other = rows(24)
            ids = r.append(other)
            r.confirm(ids, other["reward"])
            train = r.sample(True, validation=False)
            valid = r.sample(True, validation=True)
            self.assertTrue(torch.all(train["row_id"] >= 24))
            self.assertTrue(torch.all(valid["row_id"] < 24))
            state = r.state_dict()
            copy = self.make(Path(p) / "restored")
            copy.restore(state)
            self.assertTrue(
                torch.equal(copy.labels["validation"][:48], r.labels["validation"][:48])
            )
            r.close()
            copy.close()

    def test_capacity_never_overwrites(self):
        with tempfile.TemporaryDirectory() as p:
            r = self.make(p)
            r.append(rows(512))
            with self.assertRaises(OverflowError):
                r.append(rows(1))
            r.close()


class LearnerTests(unittest.TestCase):
    def test_default_actor_matches_shipped_teacher_before_updates(self):
        from algorithm.mos9.teacher import Teacher, RobotSpec, DEFAULT_POLICY

        if not DEFAULT_POLICY.exists():
            self.skipTest("workspace teacher artifact not available")
        spec = RobotSpec()
        teacher = Teacher(spec, "cpu")
        actor = Actor(512, torch.as_tensor(spec.limits))
        obs = torch.randn(32, 63)
        with torch.no_grad():
            expected = teacher.act(obs, torch.ones(32, dtype=torch.long) * 100)
            torch.testing.assert_close(actor(obs), expected)

    def test_raw_feedback_survives_target_clipping(self):
        from algorithm.mos9.teacher import RobotSpec

        with tempfile.TemporaryDirectory() as p:
            r = Replay(
                p,
                capacity=64,
                block_size=8,
                ram_blocks=8,
                batch_size=8,
                device="cpu",
                gpu_cache_bytes=0,
            )
            learner = Learner(
                r, torch.as_tensor(RobotSpec().limits), Settings(hidden=16)
            )
            model = learner.modules["actor"]
            with torch.no_grad():
                model.net[-1].weight.zero_()
                model.net[-1].bias.fill_(100)
            raw = learner.act(torch.zeros(2, 63))
            action = model.targets(raw)
            self.assertTrue(torch.all(raw == 100))
            self.assertTrue(torch.all(action <= model.high))
            reconstructed = (action - model.home) / model.scale
            self.assertGreater(float((raw - reconstructed).abs().max()), 50)
            r.close()

    def test_large_critic_error_keeps_gradient(self):
        c = Critic(16, 1000, False)
        with torch.no_grad():
            c.net[-1].bias.fill_(-100)
        loss = (c(torch.zeros(4, 63)) - 0.2).square().mean()
        loss.backward()
        self.assertGreater(c.net[-1].bias.grad.abs().min().item(), 0.1)

    def test_bellman_terminal_and_head_one(self):
        r = torch.tensor([-5.0, 0.01, 0.02])
        terminal = torch.tensor([True, False, True])
        q = torch.ones(3, 1000)
        target = bellman(r, terminal, q)
        torch.testing.assert_close(target[:, 0], r)
        self.assertTrue(torch.all(target[0] == -5))
        self.assertTrue(torch.all(target[2] == 0.02))

    def test_all_components_use_shared_sampler_and_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as p:
            r = Replay(
                p,
                capacity=64,
                block_size=8,
                ram_blocks=8,
                batch_size=8,
                device="cpu",
                gpu_cache_bytes=0,
            )
            d = rows(24)
            ids = r.append(d)
            r.confirm(ids, d["reward"])
            limits = torch.tensor([[-3.0, 3.0]] * 18)
            learner = Learner(r, limits, Settings(hidden=16))
            calls = []
            orig = r.sample

            def sample(reference=False, **kwargs):
                calls.append(reference)
                return orig(reference, **kwargs)

            r.sample = sample
            learner.fit_action_scale()
            learner.iql_update(r.sample(True))
            learner.initialize_online()
            metrics = learner.update()
            self.assertIn(True, calls)
            self.assertIn(False, calls)
            self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values()))
            new = Learner(r, limits, Settings(hidden=16))
            new.load_state_dict(learner.state_dict())
            torch.testing.assert_close(
                learner.act(d["observation"]), new.act(d["observation"])
            )
            r.close()


class DatasetTests(unittest.TestCase):
    def test_import_preserves_raw_action_and_executed_target_separately(self):
        d = rows(60)
        d["next_observation"][:, 45:] = 10
        converted = learner_transition(d)
        self.assertTrue(torch.all(converted["action"] == 10))
        self.assertTrue(torch.all(converted["executed_action"] == 0))
        self.assertTrue(converted["settling"][:50].all())
        self.assertFalse(converted["settling"][50:].any())
        self.assertFalse(converted["next_settling"][49])

    def test_existing_fp16_contract_and_episode(self):
        if not Path("runs/mos9_demos/teacher_tracking_v2_fp16/manifest.json").exists():
            self.skipTest("workspace demonstrations not available")
        data = next(episodes("runs/mos9_demos/teacher_tracking_v2_fp16"))
        self.assertEqual(data["observation"].shape[1], 63)
        self.assertTrue(data["success"][-1])
        self.assertFalse(data["terminal"][:-1].any())


if __name__ == "__main__":
    unittest.main()
