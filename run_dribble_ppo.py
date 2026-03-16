import os
import numpy as np
import genesis as gs
import torch
import torch.nn as nn

from skrl.agents.torch.ppo      import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch        import RandomMemory
from skrl.trainers.torch        import SequentialTrainer

from robots.mos9                import MOS9, MOS9Config
from robots.controlled_robot    import (ControlledRobotWrapper, 
                                        ControlledRobotWrapperConfig)
from fields.soccer_field        import SoccerField, SoccerFieldConfig
from models.mos9_walk_model     import MOS9WalkModelConfig, MOS9WalkModel
from models.dribble_model       import DribbleModelConfig, DribbleModel
from envs.dribble               import DribbleEnv,  DribbleEnvConfig
from network                    import Policy, Value


EVAL = False
DEVICE = "cuda"
RESUME_TRAINING = False
EXPERIMENT_NAME = "dribble_MOS9_v4"
CHECKPOINT_PATH = f"runs/{EXPERIMENT_NAME}/checkpoints/best_agent.pt"
NUM_ENVS = 1 if EVAL else 8192
FIELD_RANGE = 0.0
ROLLOUT_STEPS = 32


gs.init(backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True, 
        logging_level='warning')

env = DribbleEnv(DribbleEnvConfig(
    robot_cfg       = ControlledRobotWrapperConfig(
        robot_cfg      = MOS9Config(
            initial_pos     = np.array([-1.0, 0.0, 0.5], dtype=np.float32),
        ),
        robot_class     = MOS9,
        ctrl_model_cfg       = MOS9WalkModelConfig(
            n_dofs = 12, 
            target_q_offset = np.array([0.0] * 12),
        ),
        ctrl_model_class     = MOS9WalkModel,
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


models = {
    "policy": Policy(env.observation_space, env.action_space, DEVICE),
    "value": Value(env.observation_space, env.action_space, DEVICE),
}

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = ROLLOUT_STEPS
cfg["discount_factor"] = 0.97
cfg["learning_epochs"] = 8
cfg["mixed_precision"] = True
# mini_batches is the number of divisions of the total collected data (ROLLOUT_STEPS * NUM_ENVS)
cfg["mini_batches"] = 4

cfg["experiment"]["directory"] = "runs"                     # type: ignore
cfg["experiment"]["write_interval"] = 50                    # type: ignore
cfg["experiment"]["checkpoint_interval"] = 1000             # type: ignore
cfg["experiment"]["experiment_name"] = EXPERIMENT_NAME      # type: ignore

agent = PPO(models=models,
            memory=RandomMemory(memory_size=ROLLOUT_STEPS, 
                                num_envs=env.num_envs, 
                                device=DEVICE),
            cfg=cfg,
            observation_space=env.observation_space, # type: ignore
            action_space=env.action_space, # type: ignore
            device=DEVICE)

# --- Training ---

cfg_trainer = { "timesteps": 30000, 
                "headless": True,
                "environment_info": "extra", }
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent]) # type: ignore
    
if (EVAL or RESUME_TRAINING) and os.path.exists(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    print(f"Model loaded from {CHECKPOINT_PATH}")

if not EVAL:
    trainer.train()
else:
    agent.policy.eval() # type: ignore[union-attr]
    states, _ = env.reset()
    with torch.no_grad():
        for i in range(1000):
            actions, _, _ = agent.act(states, timestep=0, timesteps=0)
            # actions = states[:, 7:10] * 0.5
            # actions[:, 2] = 0.0
            (states, _, _, _, _) = env.step(actions)
