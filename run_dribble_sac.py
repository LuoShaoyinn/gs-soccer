import os
import numpy as np
import genesis as gs
import torch
import torch.nn as nn

from skrl.agents.torch.sac      import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch        import RandomMemory
from skrl.trainers.torch        import SequentialTrainer

from robots.mos9                import MOS9, MOS9Config
from robots.controlled_robot    import (ControlledRobotWrapper,
                                        ControlledRobotWrapperConfig)
from fields.soccer_field        import SoccerField, SoccerFieldConfig
from models.walk_model          import WalkModelConfig, WalkModel
from models.dribble_model       import DribbleModelConfig, DribbleModel
from envs.dribble               import DribbleEnv,  DribbleEnvConfig
from network                    import Policy, QNetwork


EVAL = True
DEVICE = "cuda"
RESUME_TRAINING = False
EXPERIMENT_NAME = "dribble_sac_1"
CHECKPOINT_PATH = f"runs/SAC_Walker/{EXPERIMENT_NAME}/checkpoints/best_agent.pt"
NUM_ENVS = 1 if EVAL else 8192

gs.init(backend=gs.gpu,
        performance_mode=True,
        logging_level='warning')


env = DribbleEnv(DribbleEnvConfig(
    robot_cfg       = ControlledRobotWrapperConfig(
        robot_cfg      = MOS9Config(
            initial_pos     = np.array([-1.0, 0.0, 0.5], dtype=np.float32),
        ),
        robot_class     = MOS9,
        ctrl_model_cfg       = WalkModelConfig(
            n_dofs = 12,
            target_q_offset = np.array([0.0] * 12),
        ),
        ctrl_model_class     = WalkModel,
        ctrl_policy_path     = "models_ckpt/walk_v3_t8.pt",
    ),
    robot_class     = ControlledRobotWrapper,
    field_cfg       = SoccerFieldConfig(),
    field_class     = SoccerField,
    model_cfg       = DribbleModelConfig(),
    model_class     = DribbleModel,
    num_envs        = NUM_ENVS,
    show_viewer     = EVAL,
    ctrl_freq_ratio = 10,
    policy_freq     = 50,
    sim_freq        = 500,
))


# -----------------------------
# Networks
# -----------------------------

models = {
    "policy": Policy(env.observation_space, env.action_space, DEVICE),

    "critic_1": QNetwork(env.observation_space, env.action_space, DEVICE),
    "critic_2": QNetwork(env.observation_space, env.action_space, DEVICE),

    "target_critic_1": QNetwork(env.observation_space, env.action_space, DEVICE),
    "target_critic_2": QNetwork(env.observation_space, env.action_space, DEVICE),
}


for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# -----------------------------
# SAC Config
# -----------------------------

cfg = SAC_DEFAULT_CONFIG.copy()

cfg["discount_factor"] = 0.97
cfg["batch_size"] = 4096
cfg["random_timesteps"] = 20000
cfg["learning_starts"] = 20000
cfg["gradient_steps"] = 8

cfg["mixed_precision"] = True

cfg["experiment"]["directory"] = "runs/SAC_Walker"          # type: ignore
cfg["experiment"]["write_interval"] = 50                    # type: ignore
cfg["experiment"]["checkpoint_interval"] = 1000             # type: ignore
cfg["experiment"]["experiment_name"] = EXPERIMENT_NAME      # type: ignore


# -----------------------------
# Replay Buffer
# -----------------------------

memory = RandomMemory(
    memory_size = 2048,
    num_envs = env.num_envs,
    device = DEVICE
)


# -----------------------------
# Agent
# -----------------------------

agent = SAC(
    models = models,
    memory = memory,
    cfg = cfg,
    observation_space = env.observation_space,
    action_space = env.action_space,
    device = DEVICE
)


# -----------------------------
# Trainer
# -----------------------------

cfg_trainer = {
    "timesteps": 30000000,
    "headless": True,
    "environment_info": "extra",
}

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])


# -----------------------------
# Resume / Eval
# -----------------------------

if (EVAL or RESUME_TRAINING) and os.path.exists(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    print(f"Model loaded from {CHECKPOINT_PATH}")


if not EVAL:
    trainer.train()

else:
    agent.policy.eval()  # type: ignore

    states, _ = env.reset()

    with torch.no_grad():
        for i in range(1000):
            actions, _, _ = agent.act(states, timestep=0, timesteps=0)
            states, _, _, _, _ = env.step(actions)
