import os
import numpy as np
import genesis as gs
import torch
import torch.nn as nn
from functools import partial

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from envs.single_walker import SingleWalkerEnv, SingleWalkerEnvConfig
from core_func import reward_fn, truncated_fn, terminated_fn, gen_cmd_fn
from network import Policy, Value

EVAL = False
RESUME_TRAINING = False
CHECKPOINT_PATH = "runs/PPO_Walker/fuck2/checkpoints/best_agent.pt"
NUM_ENVS = 16 if EVAL else 8192
DEVICE = "cuda"
FIELD_RANGE = 2.0
ROLLOUT_STEPS = 48

set_seed(42)
gs.init(backend=gs.gs_backend.vulkan, 
        performance_mode=True, 
        logging_level='info')

# Environment Setup
env = SingleWalkerEnv(SingleWalkerEnvConfig(
    num_envs        = NUM_ENVS,
    robot_URDF      = 'assets/MOS9/MOS9.urdf',
    field_range     = FIELD_RANGE, 
    reward_fn       = partial(reward_fn, num_envs=NUM_ENVS, device=DEVICE), 
    action_scale    = np.array([0.2] * 12, dtype=np.float32), 
    terminated_fn   = partial(terminated_fn, link_force_threshold=500), 
    force_range     = np.array([[-120.0] * 12, [120.0] * 12], dtype=np.float32), 
    truncated_fn    = partial(truncated_fn, field_range=FIELD_RANGE), 
    gen_cmd_fn      = partial(
        gen_cmd_fn,
        low=np.array([-0.6, -0.3, -1.0], dtype=np.float32),
        high=np.array([0.8, 0.3, 1.0], dtype=np.float32),
        device=DEVICE,
    ),
    show_viewer     = EVAL, 
    robot_initial_pos  = np.array([0.0, 0.0, 0.51]),
    robot_initial_quat = np.array([1.0, 0.0, 0.0, 0.0]),
    kp              = np.array([40.0] * 12, dtype=np.float32),
    kv              = np.array([4.0] * 12, dtype=np.float32), 
    joint_names = ['b_Rh', 'b_Lh', 'Rh_Rl', 'Lh_Ll', 'Rl_Rl1', 'Ll_Ll1', 
                   'Rl1_Rl2', 'Ll1_Ll2', 'Rl2_Ra', 'Ll2_La', 'Ra_Rf', 'La_Lf']
))

# --- PPO Setup ---

models = {
    "policy": Policy(env.observation_space, env.action_space, DEVICE),
    "value": Value(env.observation_space, env.action_space, DEVICE),
}

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = ROLLOUT_STEPS
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_epochs"] = 8
cfg["learning_rate"] = 3e-4
cfg["mixed_precision"] = True
cfg["entropy_loss_scale"] = 0.005
# mini_batches is the number of divisions of the total collected data (ROLLOUT_STEPS * NUM_ENVS)
cfg["mini_batches"] = 64
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": DEVICE}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": DEVICE}

cfg["experiment"]["directory"] = "runs/PPO_Walker" # type: ignore
cfg["experiment"]["write_interval"] = 100           # type: ignore
cfg["experiment"]["checkpoint_interval"] = 1000     # type: ignore
cfg["experiment"]["experiment_name"] = "fuck2"      # type: ignore

agent = PPO(models=models,
            memory=RandomMemory(memory_size=ROLLOUT_STEPS, 
                                num_envs=env.num_envs, 
                                device=DEVICE),
            cfg=cfg,
            observation_space=env.observation_space, # type: ignore
            action_space=env.action_space, # type: ignore
            device=DEVICE)

# --- Training ---

cfg_trainer = { "timesteps": 1000000, 
                "headless": True,
                "environment_info": "extra", 
                # "disable_progressbar": True,
               }
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent]) # type: ignore
    
if (EVAL or RESUME_TRAINING) and os.path.exists(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    print(f"Model loaded from {CHECKPOINT_PATH}")

if not EVAL:
    trainer.train()
else:
    agent.policy.eval() # type: ignore
    states, _ = env.reset()
    with torch.no_grad():
        while True: # Run indefinitely or for a fixed range
            # Use deterministic actions for evaluation
            actions, _, _ = agent.act(states, timestep=0, timesteps=0)
            env.step(actions) 
