"""URSoccerLab's learned walking policy, used as an explicit reference teacher."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from genesis.utils.geom import transform_by_quat

DEFAULT_SOURCE = Path('/home/luoshaoyinn/workspace/URSoccerLab')
DEFAULT_POLICY = DEFAULT_SOURCE/'py_example/models/policies/mos9_walk_v11_5500.onnx'


class OnnxWalkingTeacher:
    def __init__(self, env, names, kin, settings, policy=DEFAULT_POLICY,
                 actuator_model=DEFAULT_SOURCE/'external/robots/mos9/model.xml', vx=.4):
        import onnxruntime as ort
        self.env = env
        self.device = env.MDP.home.device
        self.names = list(names)
        home = np.zeros(len(names), dtype=np.float32)
        home[self.names.index('right_shoulder_roll')] = -1.4
        home[self.names.index('left_shoulder_roll')] = 1.4
        self.home = torch.as_tensor(home, device=self.device)
        env.MDP.cfg.home = home
        env.MDP.home = self.home
        env.MDP.cfg.height = (settings.terrain_height if settings.terrain == 'gentle' else 0) + settings.spawn_clearance - kin.collision_bottom(home)
        actuators = {a.get('name'): a for a in ET.parse(actuator_model).getroot().find('actuator')}
        servo_names = [('r' if n.startswith('right') else 'l')+'_'+n.split('_', 1)[1]+'_joint_servo' for n in names]
        kp = [float(actuators[n].get('kp')) for n in servo_names]
        kv = [float(actuators[n].get('kv')) for n in servo_names]
        limits = np.array([list(map(float, actuators[n].get('forcerange').split())) for n in servo_names])
        entity, indices = env.robot.robot, env.robot.dofs_idx_local
        entity.set_dofs_kp(kp, dofs_idx_local=indices)
        entity.set_dofs_kv(kv, dofs_idx_local=indices)
        entity.set_dofs_force_range(limits[:, 0], limits[:, 1], dofs_idx_local=indices)
        entity.set_dofs_armature(np.zeros(len(names)), dofs_idx_local=indices)
        small_motor = np.array(['ankle_roll' in n or 'shoulder_pitch' in n or 'shoulder_roll' in n for n in names])
        scales = .25*np.where(small_motor, 36., 60.)/np.where(small_motor, 59.59586122651323, 98.30757637604704)
        self.scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        self.lower, self.upper = torch.tensor([kin.bounds[n] for n in names], device=self.device).T
        self.previous = torch.zeros((env.num_envs, len(names)), device=self.device)
        self.command = torch.tensor([vx, 0., 0.], device=self.device).expand(env.num_envs, -1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(policy), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.metadata = dict(kind='onnx', policy=str(policy), sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
                             actuator_model=str(actuator_model), actuator_sha256=hashlib.sha256(Path(actuator_model).read_bytes()).hexdigest(),
                             kp=kp, kv=kv, torque_limits=limits.tolist(), armature=0., vx=vx,
                             observation_dim=63, standing_steps=50, standing_included_in_500_step_horizon=True)

    @torch.no_grad()
    def act(self, state, steps, origins, yaws):
        self.previous[steps == 0] = 0
        inverse = state['body_quat'].clone()
        inverse[:, 1:] *= -1
        angular = transform_by_quat(self.env.robot.robot_base.get_ang(), inverse)
        gravity = torch.zeros((self.env.num_envs, 3), device=self.device)
        gravity[:, 2] = -1
        gravity = transform_by_quat(gravity, inverse)
        obs = torch.cat((angular*.2, gravity, self.command, state['dofs_pos']-self.home,
                         state['dofs_vel']*.05, self.previous), dim=1)
        raw = self.session.run(None, {self.input_name: obs.cpu().numpy().astype(np.float32)})[0]
        self.previous.copy_(torch.as_tensor(raw, device=self.device))
        target = (self.home+self.previous*self.scales).clamp(self.lower, self.upper)
        standing = steps < 50
        self.previous[standing] = 0
        return torch.where(standing[:, None], self.home, target)

    @torch.no_grad()
    def observe_applied_action(self, action, teacher_mask):
        # If the learner controls an environment, the next expert observation
        # must describe the action that was physically applied there.
        mask = ~teacher_mask
        self.previous[mask] = (action[mask]-self.home)/self.scales
