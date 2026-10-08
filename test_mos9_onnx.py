"""Test the URSoccerLab MOS9 ONNX walker in this branch's Genesis MDP."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import genesis as gs
import onnxruntime as ort
from PIL import Image
from mos9_walk import parse_args, build_walking_env
from algorithm.mos9_teacher.environment import WalkingEnv

SOURCE = Path('/home/luoshaoyinn/workspace/URSoccerLab')

class SnapshotEnv(WalkingEnv):
    def build(self):
        super().build()
        self.camera = self.scene.add_camera(res=(960,720), pos=(1.15,-1.5,0.95),
                                            lookat=(0,0,.3), fov=35, GUI=False)

def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,default=SOURCE/'py_example/models/policies/mos9_walk_v11_5500.onnx')
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--vx',type=float,default=.4)
    p.add_argument('--output',type=Path,default=Path('runs/mos9_teacher/onnx_reference'))
    p.add_argument('--render',action='store_true')
    args,rest=p.parse_known_args()
    settings=parse_args(rest)
    if settings.num_envs != 1: p.error('reference evaluator requires num-envs=1')
    args.output.mkdir(parents=True,exist_ok=True)
    env,plan,kin,*_=build_walking_env(settings,SnapshotEnv if args.render else WalkingEnv)
    names=list(plan['names'])
    home=np.zeros(18,dtype=np.float32); home[names.index('right_shoulder_roll')]=-1.4; home[names.index('left_shoulder_roll')]=1.4
    model=ET.parse(SOURCE/'external/robots/mos9/model.xml').getroot()
    actuators={a.get('name'):a for a in model.find('actuator')}
    servo=[('r' if n.startswith('right') else 'l')+'_'+n.split('_',1)[1]+'_joint_servo' for n in names]
    kp=[float(actuators[n].get('kp')) for n in servo]; kv=[float(actuators[n].get('kv')) for n in servo]
    limits=np.array([list(map(float,actuators[n].get('forcerange').split())) for n in servo])
    entity=env.robot.robot; idx=env.robot.dofs_idx_local
    entity.set_dofs_kp(kp,dofs_idx_local=idx); entity.set_dofs_kv(kv,dofs_idx_local=idx)
    entity.set_dofs_force_range(limits[:,0],limits[:,1],dofs_idx_local=idx)
    entity.set_dofs_armature(np.zeros(18),dofs_idx_local=idx)
    scales=np.array([.25*(36 if ('ankle_roll' in n or 'shoulder_pitch' in n or 'shoulder_roll' in n) else 60)/(59.59586122651323 if ('ankle_roll' in n or 'shoulder_pitch' in n or 'shoulder_roll' in n) else 98.30757637604704) for n in names])
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.policy),sess_options=options,providers=['CPUExecutionProvider'])
    input_name=session.get_inputs()[0].name
    bounds=np.array([kin.bounds[n] for n in names])
    def snapshot(name):
        if args.render:
            pos=env.robot.robot_base.get_pos()[0].cpu().numpy()
            env.camera.set_pose(pos=pos+np.array([1.15,-1.5,.55]),
                                lookat=pos+np.array([0,0,-.18]))
            rgb=env.camera.render()[0]; Image.fromarray(rgb).save(args.output/(name+'.png'))
    snapshot('planner_initial_pose')
    env.MDP.cfg.home=home; env.MDP.home=torch.tensor(home,device=gs.device)
    env.MDP.cfg.height=(settings.terrain_height if settings.terrain=='gentle' else 0)+settings.spawn_clearance-kin.collision_bottom(home)
    env.reset(); snapshot('onnx_initial_pose')
    rows=[]; trace=[]
    try:
        for episode in range(args.episodes):
            env.reset()
            # Settle for one simulated second, matching the source example.
            cursor=len(env.MDP.completed)
            for _ in range(50):
                env.step(torch.tensor(home[None],device=gs.device))
                if len(env.MDP.completed)>cursor: break
            if len(env.MDP.completed)>cursor:
                rows.append(dict(env.MDP.completed[-1],episode=episode,stage='standing')); continue
            env.MDP.steps.zero_(); env.MDP.origins.copy_(env.get_state(env.all_envs_idx)['body_pos'])
            previous=np.zeros(18,dtype=np.float32)
            for step in range(500):
                state=env.get_state(env.all_envs_idx)
                q=state['body_quat'][0].cpu().numpy(); R=rotation(q)
                angular=env.robot.robot_base.get_ang()[0].cpu().numpy()
                obs=np.concatenate([R.T@angular*.2,R.T@np.array([0,0,-1]),[args.vx,0,0],state['dofs_pos'][0].cpu().numpy()-home,state['dofs_vel'][0].cpu().numpy()*.05,previous]).astype(np.float32)
                raw=session.run(None,{input_name:obs[None]})[0][0]
                target=np.clip(home+raw*scales,bounds[:,0],bounds[:,1]).astype(np.float32)
                previous=raw.astype(np.float32)
                if episode==0: trace.append(dict(step=step,pos=state['body_pos'][0].tolist(),quat=q.tolist(),target=target.tolist()))
                if episode==0 and step in (100,250): snapshot('walk_'+str(step))
                env.step(torch.tensor(target[None],device=gs.device))
                if (step+1)%100==0: print(f'episode={episode} steps={step+1}',flush=True)
                if len(env.MDP.completed)>cursor:
                    row=dict(env.MDP.completed[-1],episode=episode,stage='walking'); rows.append(row); print(row,flush=True); break
    finally:
        result=dict(policy=str(args.policy),sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),terrain=settings.terrain,terrain_height=settings.terrain_height,seed=settings.seed,flat_soles=settings.flat_soles,vx=args.vx,frequency=50,sim_frequency=settings.sim_freq,spawn_height=env.MDP.cfg.height,spawn_clearance=settings.spawn_clearance,settling_steps=50,kp=kp,kv=kv,torque_limits=limits.tolist(),armature=0.0,episodes=rows,trace=trace)
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        env.scene.destroy()
    print('Saved',args.output,flush=True)

if __name__=='__main__': main()
