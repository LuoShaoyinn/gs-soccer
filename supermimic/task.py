"""Shared analytic observation/reward heads used in simulation and imagination."""

import torch
from torch import nn

from .config import JOINT_NAMES, TRACKED_LINKS, URDF
from .geometry import (
    Kinematics,
    conjugate,
    gravity,
    multiply,
    normalize,
    rotate,
    rotation_6d,
    rotation_error_squared,
)


class Task(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.n = cfg, len(JOINT_NAMES)
        n = self.n
        self.q = slice(0, n)
        self.dq = slice(n, 2 * n)
        self.pos = slice(2 * n, 2 * n + 3)
        self.quat = slice(2 * n + 3, 2 * n + 7)
        self.vel = slice(2 * n + 7, 2 * n + 10)
        self.ang = slice(2 * n + 10, 2 * n + 13)
        self.previous = slice(2 * n + 13, 3 * n + 13)
        self.physical_dim, self.state_dim = 2 * n + 13, 3 * n + 13
        self.obs_dim, self.goal_frame_dim = 3 * n + 6, 2 * n + 15
        self.context_dim = n + 4
        self.kin = Kinematics(URDF, JOINT_NAMES, TRACKED_LINKS)
        self.register_buffer("home", torch.zeros(n))
        # scale: q, dq, pos, quat, base velocity, base angular velocity, previous action
        self.register_buffer(
            "scale",
            torch.cat(
                (
                    torch.ones(n),
                    torch.full((n,), 10.0),
                    torch.ones(10),
                    torch.full((3,), 5.0),
                    torch.ones(n),
                )
            ),
        )
        self.register_buffer(
            "noise",
            torch.tensor([0.01] * n + [0.5] * n + [0.2] * 3 + [0.05] * 3 + [0.0] * n),
        )

    def pack(self, q, dq, pos, quat, vel, ang, previous=None):
        if previous is None:
            previous = torch.zeros_like(q)
        return torch.cat((q, dq, pos, normalize(quat), vel, ang, previous), -1)

    def observation(self, state, noisy=False):
        quat = state[..., self.quat]
        obs = torch.cat(
            (
                state[..., self.q] - self.home,
                state[..., self.dq],
                rotate(conjugate(quat), state[..., self.ang]),
                gravity(quat),
                state[..., self.previous],
            ),
            -1,
        )
        if noisy:
            obs = obs + (2 * torch.rand_like(obs) - 1) * self.noise
        # Fixed observation scaling; no moving statistics that differ between learners.
        return torch.cat(
            (
                obs[..., : self.n],
                obs[..., self.n : 2 * self.n] * 0.1,
                obs[..., 2 * self.n : 2 * self.n + 3] * 0.2,
                obs[..., 2 * self.n + 3 :],
            ),
            -1,
        )

    def goal(self, state, reference):
        # Future targets in the current root frame. The motion interface needs a root-pose estimate.
        q = state[..., self.quat].unsqueeze(-2)
        inv = conjugate(q)
        relpos = rotate(
            inv, reference[..., self.pos] - state[..., self.pos].unsqueeze(-2)
        )
        relrot = rotation_6d(multiply(inv, reference[..., self.quat]))
        return torch.cat(
            (
                reference[..., self.q] - self.home,
                reference[..., self.dq] * 0.1,
                relpos,
                relrot,
                rotate(inv, reference[..., self.vel]),
                rotate(inv, reference[..., self.ang]) * 0.2,
            ),
            -1,
        ).flatten(-2)

    def bodies(self, state):
        return self.kin(
            *(
                state[..., s]
                for s in (self.q, self.dq, self.pos, self.quat, self.vel, self.ang)
            )
        )

    def reward(self, state, previous_action, action, reference):
        p, r, v, w = self.bodies(state)
        rp, rr, rv, rw = self.bodies(reference)
        terms = {
            "body_position": torch.exp(-(p - rp).square().sum(-1).mean(-1) / 0.3**2),
            "body_orientation": torch.exp(
                -rotation_error_squared(r, rr).mean(-1) / 0.4**2
            ),
            "body_velocity": torch.exp(-(v - rv).square().sum(-1).mean(-1)),
            "body_angular_velocity": torch.exp(
                -(w - rw).square().sum(-1).mean(-1) / 3.14**2
            ),
            "root_position": 0.5
            * torch.exp(
                -(state[..., self.pos] - reference[..., self.pos]).square().sum(-1)
                / 0.3**2
            ),
            "root_orientation": 0.5
            * torch.exp(
                -rotation_error_squared(
                    state[..., self.quat], reference[..., self.quat]
                )
                / 0.4**2
            ),
            "action_smoothness": -0.1 * (action - previous_action).square().sum(-1),
            "joint_limit": -2
            * (
                (state[..., self.q] - self.kin.upper).relu()
                + (self.kin.lower - state[..., self.q]).relu()
            ).sum(-1),
        }
        return sum(terms.values()) * self.cfg.dt, {
            k: v * self.cfg.dt for k, v in terms.items()
        }

    def terminal(self, state, reference):
        height = (state[..., self.pos][..., 2] - reference[..., self.pos][..., 2]).abs()
        tilt = (
            gravity(state[..., self.quat])[..., 2]
            - gravity(reference[..., self.quat])[..., 2]
        ).abs()
        return (height > self.cfg.height_threshold) | (tilt > self.cfg.tilt_threshold)

    def targets(self, action, context):
        # Keep the executed normalized action in replay even if a setpoint reaches a joint limit.
        target = (
            self.home + self.cfg.action_scale * action + context[..., 1 : 1 + self.n]
        )
        return target.clamp(self.kin.lower, self.kin.upper)
