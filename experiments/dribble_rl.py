import os
import numpy as np
import genesis as gs
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from envs.walker    import WalkEnv,     WalkEnvConfig
from envs.dribble   import DribbleEnv,  DribbleEnvConfig
from robots.mos9    import MOS9,        MOS9Config
from fields.field   import Field,       FieldConfig
from fields.soccer_field        import (SoccerField, 
                                        SoccerFieldConfig)
from robots.controlled_robot    import (ControlledRobotWrapper, 
                                        ControlledRobotWrapperConfig)
from network import Policy, Value


EVAL = True
DEVICE = "cuda"
RESUME_TRAINING = False
EXPERIMENT_NAME = "dribble_walk_2"
MODEL_PATH = f"models/walk_v3_t8.pt"
CHECKPOINT_PATH = f"runs/PPO_Walker/{EXPERIMENT_NAME}/checkpoints/best_agent.pt"
NUM_ENVS = 1 if EVAL else 8192
FIELD_RANGE = 0.0
ROLLOUT_STEPS = 32


gs.init(backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True, 
        logging_level='warning')

env = DribbleEnv(DribbleEnvConfig(
    robot_cfg       = ControlledRobotWrapperConfig(
        robot_cfg      = MOS9Config(
            self_collision  = False,
            initial_pos     = np.array([-1.0, 0.0, 0.5], dtype=np.float32),
        ),
        robot_class    = MOS9,
        policy_path    = MODEL_PATH,
    ),
    robot_class     = ControlledRobotWrapper,
    field_cfg       = SoccerFieldConfig(),
    field_class     = SoccerField,
    env_spacing     = FIELD_RANGE, 
    num_envs        = NUM_ENVS,
    show_viewer     = EVAL, 
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

cfg["experiment"]["directory"] = "runs/PPO_Walker"          # type: ignore
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
