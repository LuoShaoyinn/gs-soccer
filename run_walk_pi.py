import torch
import numpy as np
import genesis as gs
import torch.nn as nn

from envs.walker import WalkEnv, WalkEnvConfig
from robots.pi import PI, PIConfig
from fields.field import Field, FieldConfig
from models.pi_walk_model import PIWalkModelConfig, PIWalkModel


NUM_ENVS = 1
MODEL_PATH = "models_ckpt/pi_walk_40000.pt"


def load_policy(policy_path: str):
    try:
        return torch.jit.load(policy_path)
    except Exception:
        checkpoint = torch.load(policy_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "pi_actor_state_v1":
            raise RuntimeError(f"Unsupported policy format: {policy_path}")

        dims = checkpoint["dims"]
        state_dict = checkpoint["state_dict"]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        actor = nn.Sequential(*layers)
        actor.load_state_dict(state_dict, strict=True)
        actor.eval()
        return actor

gs.init(
    backend=gs.gpu,  # type: ignore[unsolved-attribute]
    performance_mode=True,
    logging_level="info",
)

env = WalkEnv(
    WalkEnvConfig(
        robot_cfg=PIConfig(),
        robot_class=PI,
        field_cfg=FieldConfig(),
        field_class=Field,
        model_cfg=PIWalkModelConfig(
            n_dofs=20,
            target_q_offset=np.array(
                [
                    -0.25,
                    0.0,
                    -0.25,
                    0.0,
                    0.0,
                    0.2,
                    0.0,
                    -0.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.65,
                    -1.2,
                    0.65,
                    -1.2,
                    -0.4,
                    -0.4,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        ),
        policy_freq=50,
        sim_freq=500,
        model_class=PIWalkModel,
        env_spacing=0.0,
        num_envs=NUM_ENVS,
        show_viewer=True,
    )
)

obs, info = env.reset()
actions = torch.zeros((NUM_ENVS, 20), dtype=torch.float, device=gs.device)
actor = load_policy(MODEL_PATH).to(gs.device)
while True:
    actions = actor(obs)
    obs, rew, terminated, truncated, info = env.step(actions)
