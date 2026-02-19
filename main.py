import os
import numpy as np
import genesis as gs
import torch
from functools import partial

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from envs.single_walker import SingleWalkerEnv, SingleWalkerEnvConfig
from core_func import (
    UNITREE_G1_REWARD_SCALES,
    reward_fn,
    truncated_fn,
    terminated_fn,
    gen_cmd_fn,
)
from network import Policy, Value

EVAL = True
RESUME_TRAINING = False
EXPERIMENT_NAME = "unitree_style_mos9"
CHECKPOINT_PATH = f"runs/PPO_Walker/{EXPERIMENT_NAME}/checkpoints/best_agent.pt"
NUM_ENVS = 16 if EVAL else 16384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIELD_RANGE = 2.0
SIM_DT = 0.005
CONTROL_DECIMATION = 4
ROLLOUT_STEPS = 24

set_seed(42)
backend = gs.gpu
gs.init(backend=backend, 
        performance_mode=True, 
        logging_level='warning')

# Unitree G1 reference control defaults mapped to this 12-DoF biped layout:
# [hip_yaw, hip_yaw, hip_roll, hip_roll, hip_pitch, hip_pitch,
#  knee, knee, ankle_pitch, ankle_pitch, ankle_roll, ankle_roll]
default_joint_angles = np.array(
    [0.0, 0.0, 0.0, 0.0, -0.1, -0.1, 0.3, 0.3, -0.2, -0.2, 0.0, 0.0],
    dtype=np.float32,
)
kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 40.0, 40.0, 40.0, 40.0], dtype=np.float32)
kv = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)

# Environment Setup
env = SingleWalkerEnv(SingleWalkerEnvConfig(
    num_envs        = NUM_ENVS,
    device          = DEVICE,
    robot_URDF      = 'assets/MOS9/MOS9.urdf',
    field_range     = FIELD_RANGE, 
    reward_fn       = partial(
        reward_fn,
        reward_dt=SIM_DT * CONTROL_DECIMATION,
        tracking_sigma=0.25,
        base_height_target=0.51,
        reward_scales=UNITREE_G1_REWARD_SCALES,
        only_positive_rewards=True,
        hip_indices=(1, 2, 7, 8),
    ), 
    action_scale    = np.array([0.25] * 12, dtype=np.float32), 
    terminated_fn   = partial(
        terminated_fn,
        base_contact_force_threshold=1.0,
        max_roll=0.8,
        max_pitch=1.0,
    ),
    force_range     = np.array([[-60.0] * 12, [60.0] * 12], dtype=np.float32), 
    truncated_fn    = partial(truncated_fn, field_range=FIELD_RANGE), 
    gen_cmd_fn      = partial(
        gen_cmd_fn,
        low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        device=DEVICE,
    ),
    sim_dt          = SIM_DT,
    control_decimation=CONTROL_DECIMATION,
    command_resample_time_s=10.0,
    episode_length_s=20.0,
    gait_period_s=0.8,
    command_zero_threshold=0.2,
    observation_clip=100.0,
    action_clip=100.0,
    show_viewer     = EVAL, 
    robot_initial_pos  = np.array([0.0, 0.0, 0.51]),
    robot_initial_quat = np.array([1.0, 0.0, 0.0, 0.0]),
    default_joint_angles=default_joint_angles,
    kp              = kp,
    kv              = kv,
    foot_link_names = ["Lfoot", "Rfoot"],
    penalized_contact_link_names = ["Rhip", "Lhip", "Rleg1", "Lleg1"],
    termination_contact_link_names = ["base"],
    joint_names = ['b_Rh', 'b_Lh', 'Rh_Rl', 'Lh_Ll', 'Rl_Rl1', 'Ll_Ll1', 
                   'Rl1_Rl2', 'Ll1_Ll2', 'Rl2_Ra', 'Ll2_La', 'Ra_Rf', 'La_Lf']
))

# --- PPO Setup ---

models = {
    "policy": Policy(env.observation_space, env.action_space, DEVICE),
    "value": Value(env.observation_space, env.action_space, DEVICE),
}

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = ROLLOUT_STEPS
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_epochs"] = 5
cfg["learning_rate"] = 1e-3
cfg["mixed_precision"] = False
cfg["entropy_loss_scale"] = 0.01
# mini_batches is the number of divisions of the total collected data (ROLLOUT_STEPS * NUM_ENVS)
cfg["mini_batches"] = 4
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": DEVICE}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": DEVICE}

cfg["experiment"]["directory"] = "runs/PPO_Walker" # type: ignore
cfg["experiment"]["write_interval"] = 100           # type: ignore
cfg["experiment"]["checkpoint_interval"] = 1000     # type: ignore
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
            actions, _, _ = agent.act(states, timestep=0, timesteps=0)
            states, _, _, _, _ = env.step(actions)
