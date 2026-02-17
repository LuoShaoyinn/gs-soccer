# core_func.py:
#   Define reward, terminated, truncated functions

import torch
    
@torch.no_grad()
def truncated_fn(field_range: float, body_pos, **kwargs
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return (torch.abs(body_pos[:, :2]) > field_range).any(dim=1), {}

@torch.no_grad()
def terminated_fn(link_force_threshold: float, net_contact_force, body_pos, **kwargs
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return body_pos[:, 2] < 0.25, {}
    # return net_contact_force.max(dim=1).values < -100, {}

@torch.no_grad()
def gen_cmd_fn(low, high, device) -> torch.Tensor:
    return (high - low) * torch.rand((3, ), device=device) + low

@torch.no_grad()
def reward_fn(num_envs: int, 
              device: str, 
              # dofs_force: torch.Tensor,       
              # body_quat: torch.Tensor,        # (B, 4) - Assumed (x, y, z, w)
              # body_vel: torch.Tensor,         
              body_pos: torch.Tensor,         
              cmd_vel: torch.Tensor,          
              **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # 1. Linear Velocity Tracking
    # High reward for matching the command, exponential decay for error
    # lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - body_vel[:, :2]), dim=1)
    # rew_lin_vel = torch.exp(-lin_vel_error / 0.02)

    # 2. Orientation Reward (Uprightness)
    # Since (1, 0, 0, 0) is upright, we penalize the 'x, y, z' components 
    # of the quaternion. If they are 0, the robot is perfectly level.
    # Formula: 1 - 2*(x^2 + y^2) is the projection of the local Z onto world Z
    # quat_xy = torch.sum(torch.square(body_quat[:, :2]), dim=1) # x^2 + y^2
    # rew_orient = torch.exp(-quat_xy / 0.1)

    # 3. Energy/Torque Penalty
    # Penalize high joint forces to prevent "jitter" and save virtual battery
    # joint_torque = 500 - torch.mean(torch.square(dofs_force), dim=1)\
    #                            .clip(min=0, max=500)
    # rew_torque = torch.exp(-joint_torque / 50)

    # 4. Base Height Reward
    rew_height = torch.square(body_pos[:, 2] - 0.51) # Target height of 0.5m
    rew_height = torch.exp(-rew_height / 0.01)

    # Weighted Sum
    # Adjust these coefficients based on your robot's behavior
    return ( 
        # 2.0 * rew_lin_vel       # Primary Goal
        # + 0.8 * rew_orient      # Keep it standing
        0.5 * rew_height      # Don't crouch
        # - 0.1 * rew_torque      # Small penalty for efficiency
    ) / (0.5), \
    {
        # "rew_lin_vel":    2.0 * rew_lin_vel.mean(), 
        # "rew_orient":   0.8 * rew_orient.mean(),
        "rew_height":   0.5 * rew_height.mean(),
        # "rew_torque":  -0.1 * rew_torque.mean(),
    }
