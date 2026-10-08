"""Teacher-only trials using the same physics and reset as training."""
import argparse
import json
from pathlib import Path
import torch
from mos9_walk import parse_args, build_walking_env, SimpleWalkingTeacher
from train_mos9_limit_action import TrainingEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pitch-gains',default='0.2,0.4,0.6,0.8,1,1.5,2,3')
    p.add_argument('--adaptive-phase', action='store_true')
    p.add_argument('--y-velocity-gains', default=None)
    p.add_argument('--y-position-gains', default=None)
    p.add_argument('--x-position-gains', default=None)
    p.add_argument('--hip-fractions', default=None)
    p.add_argument('--pitch-damping-gains', default=None)
    p.add_argument('--roll-gains', default=None)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--episodes-per-env',type=int,default=1)
    p.add_argument('--output',type=Path,default=Path('runs/mos9_teacher/pitch_sweep.json'))
    args, extra=p.parse_known_args()
    gains=[float(v) for v in args.pitch_gains.split(',')]
    xs=None if args.x_position_gains is None else [float(v) for v in args.x_position_gains.split(',')]
    rolls=None if args.roll_gains is None else [float(v) for v in args.roll_gains.split(',')]
    damp=None if args.pitch_damping_gains is None else [float(v) for v in args.pitch_damping_gains.split(',')]
    hips=None if args.hip_fractions is None else [float(v) for v in args.hip_fractions.split(',')]
    ys=None if args.y_position_gains is None else [float(v) for v in args.y_position_gains.split(',')]
    yvs=None if args.y_velocity_gains is None else [float(v) for v in args.y_velocity_gains.split(',')]
    width=len(xs or ys or yvs or rolls or damp or hips or gains)
    if len(gains)==1:gains*=width
    if len(gains)!=width:raise ValueError('gain list lengths differ')
    settings=parse_args(['--flat-soles','--sim-freq','4000','--num-envs',str(len(gains)),'--seed',str(args.seed),*extra])
    env,plan,kin,*_=build_walking_env(settings,TrainingEnv)
    device=env.MDP.home.device
    if args.adaptive_phase:
        from algorithm.mos9_teacher.phase import PhaseWalkingTeacher
        from algorithm.mos9_teacher.planner import GaitConfig
        from dataclasses import asdict
        teacher=PhaseWalkingTeacher(plan,device,GaitConfig(**{k:getattr(settings,k) for k in asdict(GaitConfig())}))
    else:
        teacher=SimpleWalkingTeacher(plan,device,settings.position_feedback,settings.velocity_feedback,settings.feedback,settings.feedback,settings.tracking_limit)
    teacher.pitch_gain=torch.tensor(gains,device=device)
    if xs is not None or ys is not None:
        xx=xs if xs is not None else [settings.position_feedback]*width
        yy=ys if ys is not None else [settings.position_feedback]*width
        teacher.position_gain=torch.tensor(list(zip(xx,yy)),device=device)
        teacher.velocity_gain=torch.tensor([[.15*x,.15*y] for x,y in zip(xx,yy)],device=device)
    if yvs is not None:teacher.velocity_gain[:,1]=torch.tensor(yvs,device=device)
    if hips is not None:teacher.pitch_hip_fraction=torch.tensor(hips,device=device)
    if damp is not None:teacher.pitch_damping=torch.tensor(damp,device=device)
    if rolls is not None:teacher.roll_gain=torch.tensor(rolls,device=device)
    lower,upper=torch.tensor([kin.bounds[n] for n in plan['names']],device=device).T
    counts=[0]*len(gains);rows=[];trace=[];cursor=0
    try:
        while min(counts)<args.episodes_per_env:
            state=env.get_state(env.all_envs_idx)
            action=teacher.act(state,env.MDP.steps,env.MDP.origins,env.MDP.yaws).clamp(lower,upper)
            trace.append({'steps':env.MDP.steps.tolist(),'phase':teacher.phase.tolist() if args.adaptive_phase else env.MDP.steps.tolist(),'position':state['body_pos'].tolist(),'quat':state['body_quat'].tolist(),'velocity':state['body_lin_vel'].tolist(),'action':action.tolist(),'joints':state['dofs_pos'].tolist(),'angular_velocity':state['body_ang_vel'].tolist(),'feet':state['foot_pos'].tolist()})
            with torch.no_grad(): env.step(action)
            if len(trace)%100==0:print(f'control iterations={len(trace)} phase={teacher.phase.tolist() if args.adaptive_phase else env.MDP.steps.tolist()}',flush=True)
            for row in env.MDP.completed[cursor:]:
                i=row['env_id']
                if counts[i]<args.episodes_per_env:
                    rows.append({**row,'pitch_gain':gains[i],'x_position_gain':xs[i] if xs else settings.position_feedback,'roll_gain':rolls[i] if rolls else settings.feedback,'pitch_damping':damp[i] if damp else 0,'hip_fraction':hips[i] if hips else 0,'y_position_gain':ys[i] if ys else settings.position_feedback,'y_velocity_gain':yvs[i] if yvs else (.15*ys[i] if ys else settings.velocity_feedback)});counts[i]+=1
                    print(rows[-1],flush=True)
            cursor=len(env.MDP.completed)
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'seed':args.seed,'adaptive_phase':args.adaptive_phase,'pitch_gains':gains,'rows':rows,'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(settings).items()}},indent=2)+'\n')
        args.output.with_suffix('.trace.json').write_text(json.dumps(trace)+'\n')
        env.scene.destroy()


if __name__=='__main__':main()
