"""Reusable teacher: planned joint targets plus small balance corrections."""
import numpy as np
import torch


class SimpleWalkingTeacher:
    def __init__(self, plan, device, position_gain=(1.0,6.0), velocity_gain=(0.15,0.9),
                 roll_gain=0.2, pitch_gain=0.6, tracking_limit=1.0, roll_damping=0.0, pitch_damping=0.0, pitch_hip_fraction=0.0):
        self.names = list(plan['names'])
        self.actions = torch.as_tensor(plan['actions'], device=device)
        self.pelvis = torch.as_tensor(plan['pelvis']-plan['pelvis'][0], device=device)
        self.velocity = torch.as_tensor(np.gradient(plan['pelvis'], 0.02, axis=0), device=device)
        self.roll = torch.as_tensor(plan['body_roll'], device=device)
        self.position_gain = torch.as_tensor(position_gain, dtype=torch.float32, device=device)
        self.velocity_gain = torch.as_tensor(velocity_gain, dtype=torch.float32, device=device)
        self.roll_gain, self.pitch_gain = roll_gain, pitch_gain
        self.tracking_limit = tracking_limit
        self.roll_damping, self.pitch_damping = roll_damping, pitch_damping
        self.pitch_hip_fraction = pitch_hip_fraction

    @torch.no_grad()
    def act(self, state, steps, origins, yaws, reference_speed=1.0):
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
                    + self.velocity_gain*(velocity-reference_speed*self.velocity[steps])).clamp(
                        -self.tracking_limit, self.tracking_limit)
        pitch_correction = self.pitch_gain*pitch+self.pitch_damping*state["body_ang_vel"][:, 1]
        for side, sign in [('right', 1), ('left', -1)]:
            target[:, self.names.index(side+'_hip_roll')] += tracking[:, 1]
            target[:, self.names.index(side+'_hip_pitch')] += sign*(self.pitch_hip_fraction*pitch_correction-tracking[:, 0])
            target[:, self.names.index(side+'_ankle_roll')] += self.roll_gain*(roll-self.roll[steps])+self.roll_damping*state["body_ang_vel"][:, 0]-tracking[:, 1]
            target[:, self.names.index(side+'_ankle_pitch')] += sign*((1-self.pitch_hip_fraction)*pitch_correction+tracking[:, 0])
        return target
