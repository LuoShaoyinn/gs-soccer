import numpy as np
import genesis as gs

from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG

from algorithms.algorithm import AlgorithmConfig
from algorithms.sac import SACAlgorithm
from robots.mos9 import MOS9, MOS9Config
from robots.controlled_robot import ControlledRobotWrapper, ControlledRobotWrapperConfig
from fields.soccer_field import SoccerField, SoccerFieldConfig
from models.mos9_walk_model import MOS9WalkModelConfig, MOS9WalkModel
from models.dribble_model import DribbleModelConfig, DribbleModel
from envs.dribble import DribbleEnv, DribbleEnvConfig
from network import Policy, QNetwork


EVAL = False
DEVICE = "cuda"
COMPILE = False
RESUME_TRAINING = False
EXPERIMENT_NAME = "dribble_sac_1"
CHECKPOINT_PATH = f"runs/SAC_Walker/{EXPERIMENT_NAME}/checkpoints/best_agent.pt"
NUM_ENVS = 1 if EVAL else 8192

gs.init(
    backend=gs.gpu,  # type: ignore[unsolved-attribute]
    performance_mode=True,
    logging_level="warning",
)

env_cfg = DribbleEnvConfig(
    robot_cfg=ControlledRobotWrapperConfig(
        robot_cfg=MOS9Config(
            initial_pos=np.array([-1.0, 0.0, 0.5], dtype=np.float32),
        ),
        robot_class=MOS9,
        ctrl_model_cfg=MOS9WalkModelConfig(
            n_dofs=12,
            target_q_offset=np.array([0.0] * 12),
        ),
        ctrl_model_class=MOS9WalkModel,
        ctrl_policy_path="models_ckpt/walk_v3_t8.pt",
    ),
    robot_class=ControlledRobotWrapper,
    field_cfg=SoccerFieldConfig(),
    field_class=SoccerField,
    model_cfg=DribbleModelConfig(),
    model_class=DribbleModel,
    num_envs=NUM_ENVS,
    show_viewer=EVAL,
    ctrl_freq_ratio=10,
    policy_freq=50,
    sim_freq=500,
)

env = DribbleEnv(env_cfg)
models = {
    "policy": Policy(env.observation_space, env.action_space, DEVICE),
    "critic_1": QNetwork(env.observation_space, env.action_space, DEVICE),
    "critic_2": QNetwork(env.observation_space, env.action_space, DEVICE),
    "target_critic_1": QNetwork(env.observation_space, env.action_space, DEVICE),
    "target_critic_2": QNetwork(env.observation_space, env.action_space, DEVICE),
}

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

agent_cfg = SAC_DEFAULT_CONFIG.copy()
agent_cfg["discount_factor"] = 0.97
agent_cfg["batch_size"] = 4096
agent_cfg["random_timesteps"] = 20000
agent_cfg["learning_starts"] = 20000
agent_cfg["gradient_steps"] = 8
agent_cfg["mixed_precision"] = True

agent_cfg["experiment"]["directory"] = "runs/SAC_Walker"  # type: ignore
agent_cfg["experiment"]["write_interval"] = 50  # type: ignore
agent_cfg["experiment"]["checkpoint_interval"] = 1000  # type: ignore
agent_cfg["experiment"]["experiment_name"] = EXPERIMENT_NAME  # type: ignore

algo_cfg = AlgorithmConfig(
    env_cfg=env_cfg,
    env_class=DribbleEnv,
    models=models,
    experiment_name=EXPERIMENT_NAME,
)

trainer_cfg = {
    "timesteps": 30000000,
    "headless": True,
    "environment_info": "extra",
}

algorithm = SACAlgorithm(
    cfg=algo_cfg,
    env=env,
    agent_cfg=agent_cfg,
    models=models,
    device=DEVICE,
    memory_size=65536,
    trainer_cfg=trainer_cfg,
)

algorithm.execute(
    eval_mode=EVAL,
    resume_training=RESUME_TRAINING,
    checkpoint_path=CHECKPOINT_PATH,
    compile_policy=COMPILE,
)
