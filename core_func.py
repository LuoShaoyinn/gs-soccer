# core_func.py:
#   Define reward, terminated, truncated functions

import torch
        
#            "ang_vel"   : self.imu.read().and_vel, 
#            "dofs_pos"  : dofs_pos, 
#            "dofs_vel"  : dofs_vel, 
#            "body_quat" : robot_base.get_quat(), 
#            "body_vel"  : robot_base.get_vel(), 
#            "body_pos"  : robot_base.get_pos(), 
#            "cmd_vel"   : self.command, 
#            "action"    : action,
#            "episode_time" : self.episode_time,


@torch.no_grad()
def truncated_fn(field_range: float, timeout: float, \
                 body_pos, episode_time, **kwargs, \
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out_of_field = (torch.abs(body_pos[:, :2]) > field_range).any(dim=1)
    out_of_time  = episode_time.squeeze(1) > timeout
    return torch.logical_or(out_of_field, out_of_time), {}


@torch.no_grad()
def terminated_fn(link_force_threshold: float,
                  body_pos: torch.Tensor,
                  body_quat: torch.Tensor,
                  **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Stop quickly on clear failure states to keep rollouts informative.
    fallen = body_pos[:, 2] < 0.25
    tilted = torch.sum(torch.square(body_quat[:, 1:3]), dim=1) > 0.35
    # Early training often has incidental self-contact; keep it as a penalty,
    # not a hard termination condition.
    terminated = fallen | tilted
    return terminated, {
        "terminated/fallen": fallen.float().mean(),
        "terminated/tilted": tilted.float().mean(),
    }


@torch.no_grad()
def gen_cmd_fn(low, high, device) -> torch.Tensor:
    low_t = torch.as_tensor(low, device=device, dtype=torch.float32)
    high_t = torch.as_tensor(high, device=device, dtype=torch.float32)
    return (high_t - low_t) * torch.rand((3,), device=device) + low_t


@torch.no_grad()
def reward_fn(collision_penalty_weight: float,
              body_pos: torch.Tensor,
              body_vel: torch.Tensor,
              body_quat: torch.Tensor,
              cmd_vel: torch.Tensor,
              ang_vel: torch.Tensor, 
              dofs_vel: torch.Tensor,
              net_contact_force: torch.Tensor,
              **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # 1) Track commanded planar velocity.
    lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - body_vel[:, :2]), dim=1)
    rew_lin_vel = torch.exp(-lin_vel_error / 0.25)

    # 2) Track yaw rate command (cmd_vel[:, 2]) from IMU angular velocity.
    yaw_rate_error = torch.square(cmd_vel[:, 2] - ang_vel[:, 2])
    rew_yaw = torch.exp(-yaw_rate_error / 0.25)

    # 3) Keep torso upright: for quat (w, x, y, z), x/y represent roll-pitch tilt.
    tilt = torch.sum(torch.square(body_quat[:, 1:3]), dim=1)
    rew_upright = torch.exp(-tilt / 0.05)

    # 4) Keep nominal base height.
    height_error = torch.square(body_pos[:, 2] - 0.51)
    rew_height = torch.exp(-height_error / 0.02)

    # 5) Discourage jittery, high-speed joint motion.
    dof_speed = torch.mean(torch.square(dofs_vel), dim=1)
    rew_smooth = torch.exp(-dof_speed / 25.0)

    # 6) Penalize non-foot body collisions (net_contact_force excludes last 2 links).
    # Rigid-body contacts can spike to ~1e4 N, so use a deadzone + linear ramp.
    # Apply this mostly when the torso is near-upright.
    peak_contact = net_contact_force.max(dim=1).values
    collision_penalty_raw = torch.clamp((peak_contact - 3000.0) / 7000.0, min=0.0, max=1.0)
    upright_gate = (rew_upright > 0.8).float()
    collision_penalty = collision_penalty_raw * upright_gate

    reward = (
        1.20 * rew_lin_vel
        + 0.30 * rew_yaw
        + 0.60 * rew_upright
        + 0.40 * rew_height
        + 0.20 * rew_smooth
        - collision_penalty_weight * collision_penalty
        + 0.20
    )
    reward = torch.clamp(reward, min=0.0, max=3.0)

    return reward, {
        "rew/lin_vel": rew_lin_vel.mean(),
        "rew/yaw": rew_yaw.mean(),
        "rew/upright": rew_upright.mean(),
        "rew/height": rew_height.mean(),
        "rew/smooth": rew_smooth.mean(),
        "rew/collision_penalty_raw": collision_penalty_raw.mean(),
        "rew/collision_penalty": collision_penalty.mean(),
        "rew/total": reward.mean(),
    }
