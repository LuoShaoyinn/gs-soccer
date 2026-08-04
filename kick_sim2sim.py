"""Run the reference kick actor in Genesis (sim2sim)."""

import argparse
import os
from pathlib import Path

import numpy as np

# Genesis and its kernel backend write generated SDF/compile caches.  Keep the
# sim2sim runner usable from a bind-mounted workspace whose original home cache
# may be read-only or inaccessible to the runtime user.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gs-soccer-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gs-soccer-matplotlib")

import genesis as gs

from envs import Env, EnvConfig
from fields import BallField, BallFieldConfig
from robots import KickPI, KickPIConfig
from MDPs import KickSim2SimMDP, KickSim2SimConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default="refs/kick_ball_0625/20260624_144537_from20260624_111401/exported/actor.onnx")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--viewer", action="store_true", default=True)
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    args = p.parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    robot_cfg = KickPIConfig()
    field_cfg = BallFieldConfig(ball_radius=0.07, ball_mass=0.25, ball_damping=0.05, ball_friction=0.6, ball_init_pos=np.array([0.8, 0.0, 0.07], dtype=np.float32))
    cfg = EnvConfig(robot_cfg=robot_cfg, robot_class=KickPI, field_cfg=field_cfg, field_class=BallField, MDP_cfg=KickSim2SimConfig(actor_path=str(Path(args.actor))), MDP_class=KickSim2SimMDP, policy_freq=50, sim_freq=500, num_envs=args.num_envs, show_viewer=args.viewer)
    env = Env(cfg)
    env.reset()
    for step in range(args.steps):
        state = env.get_state(env.all_envs_idx)
        action = env.MDP.policy_action(**state)
        if args.diagnostics and (step == 0 or (step + 1) % 50 == 0):
            print(
                f"step={step + 1:5d} action=[{action.min().item():+.3f}, {action.max().item():+.3f}] "
                f"body_z={state['body_pos'][0, 2].item():+.3f} "
                f"ball=({state['ball_pos'][0, 0].item():+.3f}, {state['ball_pos'][0, 1].item():+.3f})"
            )
        env.step(action)
    env.close()


if __name__ == "__main__":
    main()
