"""Reusable teacher: planned joint targets plus small balance corrections."""
import numpy as np
import torch


class SimpleWalkingTeacher:
    def __init__(self, plan, device, position_gain=2.0, velocity_gain=0.3,
                 roll_gain=0.2, pitch_gain=0.2, tracking_limit=1.0):
        self.names = list(plan['names'])
        self.actions = torch.as_tensor(plan['actions'], device=device)
        self.pelvis = torch.as_tensor(plan['pelvis']-plan['pelvis'][0], device=device)
        self.velocity = torch.as_tensor(np.gradient(plan['pelvis'], 0.02, axis=0), device=device)
        self.roll = torch.as_tensor(plan['body_roll'], device=device)
        self.position_gain, self.velocity_gain = position_gain, velocity_gain
        self.roll_gain, self.pitch_gain = roll_gain, pitch_gain
        self.tracking_limit = tracking_limit

    @torch.no_grad()
    def act(self, state, steps, origins, yaws):
        steps = steps.clamp(0, len(self.actions)-1)
        target = self.actions[steps].clone()
        q = state['body_quat']
        roll = torch.atan2(2*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3]),
                           1-2*(q[:, 1].square()+q[:, 2].square()))
        pitch = torch.asin((2*(q[:, 0]*q[:, 2]-q[:, 3]*q[:, 1])).clamp(-1, 1))
        c, s = torch.cos(yaws), torch.sin(yaws)
        def local(x):
            return torch.stack((c*x[:, 0]+s*x[:, 1], -s*x[:, 0]+c*x[:, 1]), dim=1)
        delta = local(state['body_pos'][:, :2]-origins[:, :2])
        velocity = local(state['body_lin_vel'][:, :2])
        tracking = (self.position_gain*(delta-self.pelvis[steps])
                    + self.velocity_gain*(velocity-self.velocity[steps])).clamp(
                        -self.tracking_limit, self.tracking_limit)
        for side, sign in [('right', 1), ('left', -1)]:
            target[:, self.names.index(side+'_hip_roll')] += tracking[:, 1]
            target[:, self.names.index(side+'_hip_pitch')] -= sign*tracking[:, 0]
            target[:, self.names.index(side+'_ankle_roll')] += self.roll_gain*(roll-self.roll[steps])-tracking[:, 1]
            target[:, self.names.index(side+'_ankle_pitch')] += sign*(self.pitch_gain*pitch+tracking[:, 0])
        return target
