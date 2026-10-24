"""Genesis collector; no simulator operation participates in autograd."""

import genesis as gs
import numpy as np
import torch

from robots.mos9 import MOS9, MOS9Config

from .config import JOINT_NAMES, TRACKED_LINKS, URDF
from .geometry import conjugate, rotate


class WalkingEnv:
    def __init__(self, task, library, sampler, viewer=False):
        self.task, self.library, self.sampler, self.cfg = (
            task,
            library,
            sampler,
            task.cfg,
        )
        cfg, n = self.cfg, task.n
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=cfg.substeps),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True,
                batch_links_info=True,
                batch_dofs_info=cfg.robust,
                constraint_solver=gs.constraint_solver.CG,
                integrator=gs.integrator.implicitfast,
                iterations=50,
                tolerance=1e-5,
            ),
            show_viewer=viewer,
        )
        if cfg.robust and cfg.terrain_height > 0:
            morph = gs.morphs.Terrain(
                height_field=task.terrain.grid.detach().cpu().numpy() / 0.001,
                horizontal_scale=cfg.terrain_cell,
                vertical_scale=0.001,
                pos=(-cfg.terrain_size / 2, -cfg.terrain_size / 2, 0),
            )
        else:
            morph = gs.morphs.Plane()
        self.floor = self.scene.add_entity(
            morph, material=gs.materials.Rigid(friction=1.0)
        )
        robot_cfg = MOS9Config(
            robot_URDF=str(URDF),
            joint_names=list(JOINT_NAMES),
            kp=np.full(n, cfg.kp, dtype=np.float32),
            kv=np.full(n, cfg.kd, dtype=np.float32),
            armature=np.full(n, 0.01, dtype=np.float32),
            damping=np.zeros(n, dtype=np.float32),
            force_range=np.array(
                [[-cfg.torque_limit] * n, [cfg.torque_limit] * n], dtype=np.float32
            ),
            velocity_range=np.array([[-20.0] * n, [20.0] * n], dtype=np.float32),
            decimate=False,
        )
        self.wrapper = MOS9(robot_cfg, self.scene)
        self.wrapper.build()
        self.robot = self.wrapper.robot
        self.scene.build(n_envs=cfg.num_envs, env_spacing=(2.0, 2.0))
        self.wrapper.config()
        self.robot.set_friction(1.0)
        self.indices = self.wrapper.dofs_idx_local
        self.links = [self.robot.get_link(n).idx_local for n in TRACKED_LINKS]
        self.root_link = self.robot.get_link("base_link").idx_local
        self.root_dofs = list(self.robot.joints[0].dofs_idx_local)
        if len(self.root_dofs) != 6:
            raise RuntimeError("MOS9 needs a free root with six velocity DOFs")
        self.nominal_com = self.robot.get_links_COM([self.root_link])[0, 0].clone()
        device = task.home.device
        self.context = torch.zeros(cfg.num_envs, task.context_dim, device=device)
        self.clip = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
        self.frame = torch.zeros_like(self.clip)
        self.steps = torch.zeros_like(self.clip)
        self.push_countdown = torch.zeros_like(self.clip)
        self.pushes_total = 0
        self.pushes_since_log = 0
        self.push_delta_max = 0.0
        self.previous = torch.zeros(cfg.num_envs, n, device=device)
        self.history = torch.zeros(
            cfg.num_envs, cfg.history, task.obs_dim, device=device
        )
        self.trajectories = [[] for _ in range(cfg.num_envs)]
        self.completed = []
        self.reset(torch.arange(cfg.num_envs, device=device))
        # These checks catch authored-vs-inertial frame mistakes before collecting training data.
        self.validate_kinematics()

    @torch.no_grad()
    def state(self):
        r, t = self.robot, self.task
        return t.pack(
            r.get_dofs_position(self.indices),
            r.get_dofs_velocity(self.indices),
            r.get_pos(),
            r.get_quat(),
            r.get_vel(),
            r.get_ang(),
            self.previous,
        )

    @torch.no_grad()
    def reset(self, idx, clips=None, frames=None):
        if len(idx) == 0:
            return
        if clips is None:
            clips, frames = self.sampler.sample(len(idx))
        self.scene.reset(envs_idx=idx)
        t, cfg, r = self.task, self.cfg, self.robot
        self.clip[idx], self.frame[idx], self.steps[idx] = clips, frames, 0
        context = torch.zeros(len(idx), t.context_dim, device=t.home.device)
        context[:, 0] = 1.0
        if cfg.randomize:

            def uniform(low, high, shape):
                return low + (high - low) * torch.rand(shape, device=t.home.device)

            context[:, 0] = uniform(*cfg.friction_range, (len(idx),))
            context[:, 1 : 1 + t.n] = (
                2
                * cfg.calibration_range
                * torch.rand(len(idx), t.n, device=t.home.device)
                - cfg.calibration_range
            )
            context[:, t.com_context] = (
                2 * torch.rand(len(idx), 3, device=t.home.device) - 1
            ) * torch.tensor(cfg.com_range, device=t.home.device)
            if cfg.robust:
                # Store deviations from nominal; zero appended inputs preserve
                # the flat-ground checkpoint when widening its model inputs.
                context[:, t.kp_context] = uniform(*cfg.kp_range, (len(idx), t.n)) - 1
                context[:, t.kd_context] = uniform(*cfg.kd_range, (len(idx), t.n)) - 1
                context[:, t.offset_context.start : t.offset_context.start + 2] = (
                    uniform(
                        -cfg.terrain_spawn_span / 2,
                        cfg.terrain_spawn_span / 2,
                        (len(idx), 2),
                    )
                )
        ratio = context[:, :1].expand(-1, r.n_links).contiguous()
        r.set_friction_ratio(ratio, envs_idx=idx)
        self.floor.set_friction_ratio(context[:, :1], envs_idx=idx)
        r.set_links_COM(
            (self.nominal_com + context[:, t.com_context]).unsqueeze(1),
            [self.root_link],
            envs_idx=idx,
        )
        if cfg.robust:
            r.set_dofs_kp(
                cfg.kp * (1 + context[:, t.kp_context]), self.indices, envs_idx=idx
            )
            r.set_dofs_kv(
                cfg.kd * (1 + context[:, t.kd_context]), self.indices, envs_idx=idx
            )
            if cfg.terrain_height > 0:
                ref = self.library.reference(clips, frames, context=context)[:, 0]
                p, q, _, _ = t.bodies(ref)
                feet = [TRACKED_LINKS.index(name) for name in ("Rfoot", "Lfoot")]
                context[:, t.offset_context.stop - 1] = t.terrain.spawn_lift(
                    p[:, feet], q[:, feet]
                )
        self.context[idx] = context
        ref = self.library.reference(clips, frames, context=context)[:, 0]
        r.set_dofs_position(ref[:, t.q], self.indices, envs_idx=idx, zero_velocity=True)
        r.set_pos(ref[:, t.pos], envs_idx=idx)
        r.set_quat(ref[:, t.quat], envs_idx=idx)
        # Free-root translations are world velocities; rotations are root-local velocities.
        root_velocity = torch.cat(
            (ref[:, t.vel], rotate(conjugate(ref[:, t.quat]), ref[:, t.ang])), -1
        )
        r.set_dofs_velocity(root_velocity, self.root_dofs, envs_idx=idx)
        r.set_dofs_velocity(ref[:, t.dq], self.indices, envs_idx=idx)
        self.push_countdown[idx] = self.sample_push_delay(len(idx))
        self.previous[idx] = 0
        r.control_dofs_position(
            t.targets(torch.zeros_like(self.previous), self.context), self.indices
        )
        obs = t.observation(self.state()[idx], cfg.sensor_noise)
        # One noisy reset reading repeated into history; old-episode observations never leak.
        self.history[idx] = obs[:, None].expand(-1, cfg.history, -1)
        for i in idx.tolist():
            self.trajectories[i] = []

    def sample_push_delay(self, count):
        low, high = self.cfg.push_interval_chunks
        return self.cfg.push_chunk_steps * torch.randint(
            low, high + 1, (count,), device=self.clip.device
        )

    @torch.no_grad()
    def apply_push(self, idx, replay=None, delta=None):
        """Instantaneous additive world xy/yaw velocity impulse, without pose reset."""
        if len(idx) == 0:
            return
        if delta is None:
            extent = torch.tensor(
                [
                    self.cfg.push_velocity,
                    self.cfg.push_velocity,
                    self.cfg.push_yaw_velocity,
                ],
                device=self.clip.device,
            )
            delta = (2 * torch.rand(len(idx), 3, device=self.clip.device) - 1) * extent
        # The free-root angular generalized velocities are body-local. Convert
        # the world-z angular increment before applying it.
        velocity = self.robot.get_dofs_velocity(self.root_dofs, envs_idx=idx)
        velocity[:, :2] += delta[:, :2]
        world_angular = torch.zeros(len(idx), 3, device=self.clip.device)
        world_angular[:, 2] = delta[:, 2]
        velocity[:, 3:] += rotate(
            conjugate(self.robot.get_quat(envs_idx=idx)), world_angular
        )
        self.robot.set_dofs_velocity(velocity, self.root_dofs, envs_idx=idx)
        if replay is not None:
            # A kick is external to the learned action dynamics. Cut only model
            # windows at this discontinuity, without labeling a fall or ending
            # the physical episode. The next recorded state contains the kick.
            replay.cut_episodes(idx)
        self.history[idx, -1] = self.task.observation(
            self.state()[idx], self.cfg.sensor_noise
        )
        self.pushes_total += len(idx)
        self.pushes_since_log += len(idx)
        self.push_delta_max = max(
            self.push_delta_max, float(delta[:, :2].norm(dim=-1).max())
        )
        self.push_countdown[idx] = self.sample_push_delay(len(idx))

    def diagnostics(self):
        t = self.task
        result = {
            "perturbation/pushes_total": self.pushes_total,
            "perturbation/pushes_this_iteration": self.pushes_since_log,
            "perturbation/max_delta_xy_speed": self.push_delta_max,
            "domain/friction_min": float(self.context[:, 0].min()),
            "domain/friction_max": float(self.context[:, 0].max()),
        }
        if self.cfg.robust:
            for label, section in (("kp", t.kp_context), ("kd", t.kd_context)):
                result[f"domain/{label}_ratio_min"] = float(
                    (1 + self.context[:, section]).min()
                )
                result[f"domain/{label}_ratio_max"] = float(
                    (1 + self.context[:, section]).max()
                )
            for i, axis in enumerate("xyz"):
                result[f"domain/com_{axis}_abs_max"] = float(
                    self.context[:, t.com_context][:, i].abs().max()
                )
        self.pushes_since_log = 0
        return result

    @torch.no_grad()
    def validate_kinematics(self):
        predicted = self.task.bodies(self.state())
        actual = (
            self.robot.get_links_pos(self.links),
            self.robot.get_links_quat(self.links),
            self.robot.get_links_vel(self.links),
            self.robot.get_links_ang(self.links),
        )
        from .geometry import rotation_error_squared

        errors = [
            float((predicted[0] - actual[0]).abs().max()),
            float(rotation_error_squared(predicted[1], actual[1]).sqrt().max()),
            float((predicted[2] - actual[2]).abs().max()),
            float((predicted[3] - actual[3]).abs().max()),
        ]
        if max(errors) > 1e-3:
            raise RuntimeError(f"MOS9 authored-frame FK/velocity mismatch: {errors}")
        return errors

    @torch.no_grad()
    def step(self, actor, replay=None, deterministic=False, auto_reset=True):
        t, cfg = self.task, self.cfg
        if cfg.push_enabled:
            self.push_countdown -= 1
            self.apply_push((self.push_countdown <= 0).nonzero().flatten(), replay)
        state = self.state()
        ref = self.library.reference(
            self.clip, self.frame, cfg.future, context=self.context
        )
        action = actor(self.history, t.goal(state, ref))
        if not deterministic:
            action = action + cfg.action_noise * torch.randn_like(action)
        if not torch.isfinite(action).all():
            raise FloatingPointError("nonfinite policy action")
        self.robot.control_dofs_position(t.targets(action, self.context), self.indices)
        self.scene.step()
        self.previous = action.clone()
        next_state = self.state()
        if not torch.isfinite(next_state).all():
            raise FloatingPointError("nonfinite Genesis state")
        target = self.library.reference(
            self.clip, self.frame + 1, context=self.context
        )[:, 0]
        reward, terms = t.reward(next_state, state[:, t.previous], action, target)
        terminal = t.terminal(next_state, target)
        self.steps += 1
        truncated = (self.frame + 1 >= self.library.lengths[self.clip] - 1) | (
            self.steps >= cfg.episode_steps
        )
        done = terminal | truncated
        if replay is not None:
            replay.append(
                state=state,
                next_state=next_state,
                action=action,
                terminal=terminal,
                done=done,
                clip=self.clip,
                frame=self.frame,
                context=self.context,
                history=self.history,
            )
        # Log factual rewards, never a reset observation substituted for next_state.
        # Copy vectors once; testing thousands of GPU scalar booleans one by one
        # otherwise synchronizes the device once per environment per step.
        for i, (c, f, rew, ended, failed, steps) in enumerate(
            zip(
                self.clip.tolist(),
                self.frame.tolist(),
                reward.tolist(),
                done.tolist(),
                terminal.tolist(),
                self.steps.tolist(),
            )
        ):
            self.trajectories[i].append((c, f, rew))
            if ended:
                if not deterministic:
                    self.sampler.update(self.trajectories[i])
                self.completed.append(
                    {
                        "clip": self.library.names[c],
                        "steps": len(self.trajectories[i]),
                        "return": sum(x[2] for x in self.trajectories[i]),
                        "failed": failed,
                        "timeout": steps >= cfg.episode_steps,
                    }
                )
        self.frame += 1
        self.history = torch.cat(
            (self.history[:, 1:], t.observation(next_state, cfg.sensor_noise)[:, None]),
            1,
        )
        if auto_reset:
            self.reset(done.nonzero().flatten())
        return reward, terminal, truncated, terms

    def close(self):
        self.scene.destroy()
