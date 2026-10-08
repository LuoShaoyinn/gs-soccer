"""Branch-local walking physics, complete reset, and terminal observations."""
import torch
import genesis as gs
from envs.env import Env


class WalkingEnv(Env):
    """Preserve physical terminal observations before automatic reset."""
    def __init__(self, cfg):
        # Keep the shared Env unchanged; configure only this experiment scene.
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_agents = 1
        self.is_vector_env = True
        self.all_envs_idx = torch.arange(cfg.num_envs, dtype=torch.long, device=gs.device)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1/cfg.policy_freq,
                                             substeps=cfg.sim_freq//cfg.policy_freq),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=cfg.self_collision,
                integrator=gs.integrator.implicitfast,
                constraint_solver=gs.constraint_solver.CG,
                iterations=50, tolerance=1e-5,
                max_collision_pairs=cfg.max_collision_pairs,
                multiplier_collision_broad_phase=cfg.multiplier_collision_broad_phase),
            show_viewer=cfg.show_viewer)
        self.build()
        self.scene.build(n_envs=cfg.num_envs, env_spacing=(cfg.env_spacing,cfg.env_spacing))
        self.config()

    def reset(self, envs_idx=None):
        idx = self.all_envs_idx if envs_idx is None else envs_idx
        # Reset solver/contact/sensor history as well as robot pose and velocity.
        self.scene.reset(envs_idx=idx)
        return super().reset(idx)

    def step(self, action):
        action = self.MDP.preprocess_action(action)
        self.robot.step(action=action)
        self.gs_step()
        state = self.get_state(self.all_envs_idx)
        obs = self.MDP.build_observation(envs_idx=self.all_envs_idx, **state)
        reward = self.MDP.build_reward(envs_idx=self.all_envs_idx, **state)
        terminated = self.MDP.build_terminated(envs_idx=self.all_envs_idx, **state)
        truncated = self.MDP.build_truncated(envs_idx=self.all_envs_idx, **state)
        self.MDP.build_info(envs_idx=self.all_envs_idx, **state)
        final_obs = obs.clone()
        done = (terminated | truncated).flatten()
        if done.any():
            idx = done.nonzero().flatten()
            obs[idx] = self.reset(idx)[0]
        return obs, reward, terminated, truncated, {"final_observation": final_obs}

