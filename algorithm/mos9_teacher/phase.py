"""Wait in double support until the physical pelvis catches up to the plan."""
import torch
from .simple import SimpleWalkingTeacher


class PhaseWalkingTeacher(SimpleWalkingTeacher):
    def __init__(self, plan, device, gait, **kwargs):
        super().__init__(plan,device,**kwargs)
        self.start = round(gait.start_time*50)
        self.period = round(gait.step_time*50)
        self.gate = round(gait.double_support*self.period/2)
        self.phase = None

    @torch.no_grad()
    def act(self,state,steps,origins,yaws):
        if self.phase is None or len(self.phase)!=len(steps):
            self.phase=torch.zeros_like(steps)
            self.speed=torch.zeros((len(steps),1),device=steps.device)
        self.phase[steps==0]=0
        self.speed[steps==0]=0
        index=self.phase.clone()
        delta=state['body_pos'][:,:2]-origins[:,:2]
        c,s=torch.cos(yaws),torch.sin(yaws)
        lateral=-s*delta[:,0]+c*delta[:,1]
        q=state['body_quat']
        roll=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        pitch=torch.asin((2*(q[:,0]*q[:,2]-q[:,3]*q[:,1])).clamp(-1,1))
        boundary=(index>=self.start)&((index-self.start)%self.period==self.gate)
        wait=boundary & ((lateral-self.pelvis[index,1]).abs()>.015)
        wait |= boundary & ((roll-self.roll[index]).abs()>.15)
        wait |= boundary & (pitch.abs()>.15)
        self.speed.lerp_((~wait).float()[:,None], .2)
        action=super().act(state,index,origins,yaws,reference_speed=self.speed)
        self.phase += (~wait).long()
        self.phase.clamp_(max=len(self.actions)-1)
        return action
