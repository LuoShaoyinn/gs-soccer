"""Closed-loop Cartesian gait tracking through Genesis GPU leg IK."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from genesis.utils.geom import transform_quat_by_quat, transform_by_quat


def quat_mul(a, b):
    return transform_quat_by_quat(b, a)


class WalkingTeacher:
    def __init__(self, robot, plan, position_gains, velocity_gains, attitude_gains, angular_damping):
        self.robot = robot
        self.names = list(plan["names"])
        self.home = torch.as_tensor(plan["home"], device=position_gains.device)
        self.device = position_gains.device
        self.position_gains = position_gains
        self.velocity_gains = velocity_gains
        self.attitude_gains = attitude_gains
        self.angular_damping = angular_damping
        self.pelvis = torch.as_tensor(plan["pelvis"], device=self.device)
        self.velocity = torch.as_tensor(np.gradient(plan["pelvis"], 0.02, axis=0), device=self.device)
        self.roll = torch.as_tensor(plan["body_roll"], device=self.device)
        self.height = float(plan["height"])
        self.feet = torch.as_tensor(plan["feet"], device=self.device).clone()
        self.feet[:, :, 2] += self.height
        from .planner import Kinematics
        kin = Kinematics()
        transforms = kin.forward(np.zeros(len(self.names)))
        self.foot_quats = [torch.as_tensor(Rotation.from_matrix(transforms[name][:3, :3]).as_quat()[[3, 0, 1, 2]],
                                           dtype=torch.float32, device=self.device) for name in ("Rfoot", "Lfoot")]
        self.q_indices = [robot.get_joint(name).qs_idx_local[0] for name in self.names]
        self.dof_indices = [robot.get_joint(name).dofs_idx_local[0] for name in self.names if "hip" in name or "knee" in name or "ankle" in name]
        self.links = [robot.get_link(name) for name in ("Rfoot", "Lfoot")]
        self.max_error = 0.0
        self.checked_state_preservation = False

    @torch.no_grad()
    def act(self, state, steps, origins, yaws, roll, pitch):
        n = len(steps)
        yaw_quat = torch.stack((torch.cos(yaws/2), torch.zeros_like(yaws), torch.zeros_like(yaws), torch.sin(yaws/2)), dim=1)
        delta_world = state["body_pos"][:, :2]-origins[:, :2]
        c, s = torch.cos(yaws), torch.sin(yaws)
        delta = torch.stack((c*delta_world[:, 0]+s*delta_world[:, 1], -s*delta_world[:, 0]+c*delta_world[:, 1]), dim=1)
        velocity_world = state["body_lin_vel"][:, :2]
        velocity = torch.stack((c*velocity_world[:, 0]+s*velocity_world[:, 1], -s*velocity_world[:, 0]+c*velocity_world[:, 1]), dim=1)
        reference = self.pelvis[steps]-self.pelvis[0]
        correction = (self.position_gains[:, None]*(delta-reference)+self.velocity_gains[:, None]*(velocity-self.velocity[steps])).clamp(-0.035, 0.035)
        command = reference-correction
        command_world = torch.stack((c*command[:, 0]-s*command[:, 1], s*command[:, 0]+c*command[:, 1]), dim=1)
        q = self.robot.get_qpos().clone()
        for column, (index, name) in enumerate(zip(self.q_indices, self.names)):
            if "shoulder" in name or "elbow" in name:
                q[:, index] = self.home[column]
        q[:, :2] = origins[:, :2]+command_world
        q[:, 2] = self.height
        commanded_roll = self.roll[steps]-(self.attitude_gains*(roll-self.roll[steps])+self.angular_damping*state["body_ang_vel"][:, 0]).clamp(-0.15, 0.15)
        commanded_pitch = -(self.attitude_gains*pitch+self.angular_damping*state["body_ang_vel"][:, 1]).clamp(-0.15, 0.15)
        rq = torch.stack((torch.cos(commanded_roll/2), torch.sin(commanded_roll/2), torch.zeros_like(commanded_roll), torch.zeros_like(commanded_roll)), dim=1)
        pq = torch.stack((torch.cos(commanded_pitch/2), torch.zeros_like(commanded_pitch), torch.sin(commanded_pitch/2), torch.zeros_like(commanded_pitch)), dim=1)
        q[:, 3:7] = quat_mul(yaw_quat, quat_mul(pq, rq))
        goals = self.feet[steps].clone()
        goals[:, :, :2] -= self.pelvis[0]
        goals = transform_by_quat(goals, yaw_quat[:, None, :].expand(-1, 2, -1))
        goals[:, :, :2] += origins[:, None, :2]
        quats = [quat_mul(yaw_quat, foot.repeat(n, 1)) for foot in self.foot_quats]
        before = self.robot.get_qpos().clone() if not self.checked_state_preservation else None
        solution, error = self.robot.inverse_kinematics_multilink(
            links=self.links, poss=[goals[:, 0], goals[:, 1]], quats=quats, init_qpos=q,
            dofs_idx_local=self.dof_indices, max_samples=1, max_solver_iters=20,
            damping=0.005, pos_tol=0.0002, rot_tol=0.002, return_error=True)
        self.max_error = max(self.max_error, float(error.abs().max()))
        if before is not None:
            if not torch.allclose(before, self.robot.get_qpos(), atol=1e-6, rtol=0):
                raise RuntimeError("IK changed the physical robot pose")
            self.checked_state_preservation = True
        # Root coordinates are used only inside IK. Execute joint targets only.
        return solution[:, self.q_indices]
