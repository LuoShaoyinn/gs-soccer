import torch


UNITREE_G1_REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "base_height": -10.0,
    "dof_acc": -2.5e-7,
    "dof_vel": -1.0e-3,
    "collision": 0.0,
    "action_rate": -0.01,
    "dof_pos_limits": -5.0,
    "alive": 0.15,
    "hip_pos": -1.0,
    "contact_no_vel": -0.2,
    "feet_swing_height": -20.0,
    "contact": 0.18,
}


@torch.no_grad()
def truncated_fn(
    field_range: float,
    body_pos: torch.Tensor,
    episode_length_buf: torch.Tensor,
    max_episode_length: torch.Tensor | int,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del kwargs
    if isinstance(max_episode_length, torch.Tensor):
        max_len = int(max_episode_length.item())
    else:
        max_len = int(max_episode_length)
    out_of_field = (torch.abs(body_pos[:, :2]) > field_range).any(dim=1)
    timeout = episode_length_buf >= max_len
    truncated = out_of_field | timeout
    return truncated, {
        "truncated/out_of_field": out_of_field.float().mean(),
        "truncated/timeout": timeout.float().mean(),
    }


@torch.no_grad()
def terminated_fn(
    base_contact_force_threshold: float,
    max_roll: float,
    max_pitch: float,
    termination_contact_force: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del kwargs
    base_contact = torch.any(
        torch.norm(termination_contact_force, dim=2) > base_contact_force_threshold, dim=1
    )
    tilted = (torch.abs(roll) > max_roll) | (torch.abs(pitch) > max_pitch)
    invalid = (~torch.isfinite(body_pos).all(dim=1)) | (~torch.isfinite(body_quat).all(dim=1))
    terminated = base_contact | tilted | invalid
    return terminated, {
        "terminated/base_contact": base_contact.float().mean(),
        "terminated/tilted": tilted.float().mean(),
        "terminated/invalid": invalid.float().mean(),
    }


@torch.no_grad()
def gen_cmd_fn(low, high, device, num_envs: int = 1) -> torch.Tensor:
    low_t = torch.as_tensor(low, device=device, dtype=torch.float32)
    high_t = torch.as_tensor(high, device=device, dtype=torch.float32)
    if low_t.numel() == 1:
        low_t = low_t.repeat(3)
    if high_t.numel() == 1:
        high_t = high_t.repeat(3)
    return (high_t - low_t).unsqueeze(0) * torch.rand((num_envs, 3), device=device) + low_t.unsqueeze(0)


@torch.no_grad()
def reward_fn(
    reward_dt: float,
    tracking_sigma: float,
    base_height_target: float,
    reward_scales: dict[str, float] | None,
    only_positive_rewards: bool,
    hip_indices: tuple[int, ...],
    body_pos: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    cmd_vel: torch.Tensor,
    dofs_pos: torch.Tensor,
    dofs_vel: torch.Tensor,
    last_dofs_vel: torch.Tensor,
    dof_pos_limits: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    feet_pos: torch.Tensor,
    feet_vel: torch.Tensor,
    feet_contact_force: torch.Tensor,
    penalized_contact_force: torch.Tensor,
    phase: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del kwargs
    scales = UNITREE_G1_REWARD_SCALES if reward_scales is None else reward_scales

    tracking_lin_error = torch.sum(torch.square(cmd_vel[:, :2] - base_lin_vel[:, :2]), dim=1)
    rew_tracking_lin = torch.exp(-tracking_lin_error / tracking_sigma)

    tracking_ang_error = torch.square(cmd_vel[:, 2] - base_ang_vel[:, 2])
    rew_tracking_ang = torch.exp(-tracking_ang_error / tracking_sigma)

    pen_lin_vel_z = torch.square(base_lin_vel[:, 2])
    pen_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
    pen_orientation = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    pen_base_height = torch.square(body_pos[:, 2] - base_height_target)
    pen_dof_acc = torch.sum(torch.square((last_dofs_vel - dofs_vel) / reward_dt), dim=1)
    pen_dof_vel = torch.sum(torch.square(dofs_vel), dim=1)
    pen_action_rate = torch.sum(torch.square(last_action - action), dim=1)

    lower = dof_pos_limits[0]
    upper = dof_pos_limits[1]
    out_of_limits = -(dofs_pos - lower).clip(max=0.0)
    out_of_limits += (dofs_pos - upper).clip(min=0.0)
    pen_dof_pos_limits = torch.sum(out_of_limits, dim=1)

    pen_collision = torch.sum((torch.norm(penalized_contact_force, dim=-1) > 0.1).float(), dim=1)
    rew_alive = torch.ones_like(rew_tracking_lin)

    valid_hip_indices = [idx for idx in hip_indices if idx < dofs_pos.shape[1]]
    if len(valid_hip_indices):
        hip_idx_t = torch.as_tensor(valid_hip_indices, dtype=torch.long, device=dofs_pos.device)
        pen_hip_pos = torch.sum(torch.square(dofs_pos.index_select(1, hip_idx_t)), dim=1)
    else:
        pen_hip_pos = torch.zeros_like(rew_tracking_lin)

    feet_contact = torch.norm(feet_contact_force, dim=2) > 1.0
    contact_feet_vel = feet_vel * feet_contact.unsqueeze(-1).float()
    pen_contact_no_vel = torch.sum(torch.square(contact_feet_vel[:, :, :3]), dim=(1, 2))
    pen_feet_swing_height = torch.sum(torch.square(feet_pos[:, :, 2] - 0.08) * (~feet_contact).float(), dim=1)

    if feet_contact.shape[1] == 2:
        phase_left = phase
        phase_right = (phase + 0.5) % 1.0
        leg_phase = torch.stack((phase_left, phase_right), dim=1)
    else:
        leg_phase = phase.unsqueeze(1).repeat(1, feet_contact.shape[1])
    is_stance = leg_phase < 0.55
    rew_contact = torch.sum((~(feet_contact ^ is_stance)).float(), dim=1)

    reward = torch.zeros_like(rew_tracking_lin)
    reward += reward_dt * scales.get("tracking_lin_vel", 0.0) * rew_tracking_lin
    reward += reward_dt * scales.get("tracking_ang_vel", 0.0) * rew_tracking_ang
    reward += reward_dt * scales.get("lin_vel_z", 0.0) * pen_lin_vel_z
    reward += reward_dt * scales.get("ang_vel_xy", 0.0) * pen_ang_vel_xy
    reward += reward_dt * scales.get("orientation", 0.0) * pen_orientation
    reward += reward_dt * scales.get("base_height", 0.0) * pen_base_height
    reward += reward_dt * scales.get("dof_acc", 0.0) * pen_dof_acc
    reward += reward_dt * scales.get("dof_vel", 0.0) * pen_dof_vel
    reward += reward_dt * scales.get("collision", 0.0) * pen_collision
    reward += reward_dt * scales.get("action_rate", 0.0) * pen_action_rate
    reward += reward_dt * scales.get("dof_pos_limits", 0.0) * pen_dof_pos_limits
    reward += reward_dt * scales.get("alive", 0.0) * rew_alive
    reward += reward_dt * scales.get("hip_pos", 0.0) * pen_hip_pos
    reward += reward_dt * scales.get("contact_no_vel", 0.0) * pen_contact_no_vel
    reward += reward_dt * scales.get("feet_swing_height", 0.0) * pen_feet_swing_height
    reward += reward_dt * scales.get("contact", 0.0) * rew_contact

    if only_positive_rewards:
        reward = torch.clamp(reward, min=0.0)

    return reward, {
        "rew/tracking_lin_vel": rew_tracking_lin.mean(),
        "rew/tracking_ang_vel": rew_tracking_ang.mean(),
        "rew/lin_vel_z": pen_lin_vel_z.mean(),
        "rew/ang_vel_xy": pen_ang_vel_xy.mean(),
        "rew/orientation": pen_orientation.mean(),
        "rew/base_height": pen_base_height.mean(),
        "rew/dof_acc": pen_dof_acc.mean(),
        "rew/dof_vel": pen_dof_vel.mean(),
        "rew/action_rate": pen_action_rate.mean(),
        "rew/dof_pos_limits": pen_dof_pos_limits.mean(),
        "rew/collision": pen_collision.mean(),
        "rew/hip_pos": pen_hip_pos.mean(),
        "rew/contact_no_vel": pen_contact_no_vel.mean(),
        "rew/feet_swing_height": pen_feet_swing_height.mean(),
        "rew/contact": rew_contact.mean(),
        "rew/alive": rew_alive.mean(),
        "rew/total": reward.mean(),
    }
