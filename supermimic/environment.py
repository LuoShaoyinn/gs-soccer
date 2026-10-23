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
                constraint_solver=gs.constraint_solver.CG,
                integrator=gs.integrator.implicitfast,
                iterations=50,
                tolerance=1e-5,
            ),
            show_viewer=viewer,
        )
        self.floor = self.scene.add_entity(
            gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0)
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
            context[:, 0] = 0.3 + 0.9 * torch.rand(len(idx), device=t.home.device)
            context[:, 1 : 1 + t.n] = (
                0.02 * torch.rand(len(idx), t.n, device=t.home.device) - 0.01
            )
            context[:, -3:] = (
                2 * torch.rand(len(idx), 3, device=t.home.device) - 1
            ) * torch.tensor([0.025, 0.05, 0.05], device=t.home.device)
        self.context[idx] = context
        ratio = context[:, :1].expand(-1, r.n_links).contiguous()
        r.set_friction_ratio(ratio, envs_idx=idx)
        self.floor.set_friction_ratio(context[:, :1], envs_idx=idx)
        r.set_links_COM(
            (self.nominal_com + context[:, -3:]).unsqueeze(1),
            [self.root_link],
            envs_idx=idx,
        )
        ref = self.library.reference(clips, frames)[:, 0]
        r.set_dofs_position(ref[:, t.q], self.indices, envs_idx=idx, zero_velocity=True)
        r.set_pos(ref[:, t.pos], envs_idx=idx)
        r.set_quat(ref[:, t.quat], envs_idx=idx)
        # Free-root translations are world velocities; rotations are root-local velocities.
        root_velocity = torch.cat(
            (ref[:, t.vel], rotate(conjugate(ref[:, t.quat]), ref[:, t.ang])), -1
        )
        r.set_dofs_velocity(root_velocity, self.root_dofs, envs_idx=idx)
        r.set_dofs_velocity(ref[:, t.dq], self.indices, envs_idx=idx)
        self.previous[idx] = 0
        r.control_dofs_position(
            t.targets(torch.zeros_like(self.previous), self.context), self.indices
        )
        obs = t.observation(self.state()[idx], cfg.sensor_noise)
        # One noisy reset reading repeated into history; old-episode observations never leak.
        self.history[idx] = obs[:, None].expand(-1, cfg.history, -1)
        for i in idx.tolist():
            self.trajectories[i] = []

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
        state = self.state()
        ref = self.library.reference(self.clip, self.frame, cfg.future)
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
        target = self.library.reference(self.clip, self.frame + 1)[:, 0]
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
        for i, (c, f, rew) in enumerate(
            zip(self.clip.tolist(), self.frame.tolist(), reward.tolist())
        ):
            self.trajectories[i].append((c, f, rew))
            if done[i]:
                if not deterministic:
                    self.sampler.update(self.trajectories[i])
                self.completed.append(
                    {
                        "clip": self.library.names[c],
                        "steps": len(self.trajectories[i]),
                        "return": sum(x[2] for x in self.trajectories[i]),
                        "failed": bool(terminal[i]),
                        "timeout": bool(self.steps[i] >= cfg.episode_steps),
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
