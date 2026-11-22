"""CPU Double DQN: shared observations, uniform replay, frozen targets."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tron_ai.observations import observation_shape

ARCHITECTURE='global-local-cnn-v1'

@dataclass(frozen=True)
class DQNConfig:
    lr:float=.0005
    gamma:float=.99
    capacity:int=16384
    batch_size:int=64
    warmup:int=512
    update_every:int=4
    target_every:int=250
    epsilon_start:float=1.
    epsilon_end:float=.05
    epsilon_decay:int=30000
    def __post_init__(self):
        if not (math.isfinite(self.lr) and self.lr>0 and 0<=self.gamma<=1 and 1<=self.batch_size<=self.warmup<=self.capacity and self.update_every>=1 and self.target_every>=1 and 0<=self.epsilon_end<=self.epsilon_start<=1 and self.epsilon_decay>=1):
            raise ValueError('invalid DQN config')

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_net=nn.Sequential(nn.Conv2d(13,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,24,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(24,32,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten())
        self.local_net=nn.Sequential(nn.Flatten(),nn.Linear(13*25,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(576,128),nn.ReLU(),nn.Linear(128,3))
    def features(self,x):
        b,c,h,w=x.shape
        pos=x[:,2].reshape(b,-1).argmax(dim=1)
        offsets=torch.arange(5,device=x.device)
        rows=pos[:,None,None]//w+offsets[None,:,None]
        cols=pos[:,None,None]%w+offsets[None,None,:]
        padded=F.pad(x,(2,2,2,2))
        padded[:,12,:2,:]=1; padded[:,12,-2:,:]=1
        padded[:,12,:,:2]=1; padded[:,12,:,-2:]=1
        patch=padded.permute(0,2,3,1)[torch.arange(b,device=x.device)[:,None,None],rows,cols].permute(0,3,1,2)
        return torch.cat((self.global_net(x),self.local_net(patch)),dim=1)
    def forward(self,x):
        return self.head(self.features(x))

class Replay:
    def __init__(self,shape,capacity):
        self.shape=tuple(shape); self.capacity=capacity; self.bits=math.prod(shape)
        n=(self.bits+7)//8
        self.obs=np.empty((capacity,n),np.uint8); self.next_obs=np.empty_like(self.obs)
        self.actions=np.empty(capacity,np.int64); self.rewards=np.empty(capacity,np.float32)
        self.done=np.empty(capacity,np.bool_); self.pos=self.size=0
    def add(self,t):
        i=self.pos
        self.obs[i]=np.packbits(t.observation.reshape(-1)); self.next_obs[i]=np.packbits(t.next_observation.reshape(-1))
        self.actions[i]=int(t.action); self.rewards[i]=t.reward; self.done[i]=t.done
        self.pos=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def sample(self,n,rng):
        if not 1<=n<=self.size: raise ValueError('invalid sample size')
        idx=rng.choice(self.size,n,replace=False)
        def unpack(a): return torch.from_numpy(np.unpackbits(a[idx],axis=1,count=self.bits).reshape(n,*self.shape)).float()
        return unpack(self.obs),torch.from_numpy(self.actions[idx]),torch.from_numpy(self.rewards[idx]),unpack(self.next_obs),torch.from_numpy(self.done[idx])

@torch.no_grad()
def double_targets(online,target,next_obs,rewards,done,gamma):
    a=online(next_obs).argmax(1,keepdim=True)
    v=target(next_obs).gather(1,a).squeeze(1)
    return rewards+gamma*(~done).float()*v

class DQNActor:
    def __init__(self,weights,config,seed):
        with torch.random.fork_rng(devices=[]): self.network=QNetwork()
        state=self.network.state_dict()
        if set(state)!=set(weights) or any(state[k].shape!=weights[k].shape or not np.isfinite(weights[k]).all() for k in state): raise ValueError('invalid DQN weights')
        self.network.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
        self.network.eval(); self.network.requires_grad_(False)
        self.rng=np.random.default_rng(seed); self.shape=observation_shape(config)
    def act(self,obs):
        if obs.shape!=self.shape: raise ValueError('observation shape mismatch')
        with torch.inference_mode(): q=self.network(torch.from_numpy(obs).float().unsqueeze(0))[0].numpy()
        return int(self.rng.choice(np.flatnonzero(q==q.max())))

def factory(weights,config,seed):
    if 'advantage.weight' in weights:
        from .advanced_dqn import factory as upgraded_factory
        return upgraded_factory(weights,config,seed)
    return DQNActor(weights,config,seed)

class DQNLearner:
    algorithm='dqn'
    def __init__(self,board,seed=2026,cfg=None,initial_weights=None):
        self.cfg=cfg or DQNConfig(); self.rng=np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.online=QNetwork(); self.target=QNetwork()
        if initial_weights is not None: self.online.load_state_dict(factory(initial_weights,board,seed).network.state_dict())
        self.target.load_state_dict(self.online.state_dict()); self.target.eval(); self.target.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.cfg.lr)
        self.replay=Replay(observation_shape(board),self.cfg.capacity)
        self.steps=self.updates=0; self.loss_sum=0.; self.loss_count=0
    @property
    def epsilon(self):
        p=min(1.,self.steps/self.cfg.epsilon_decay)
        return self.cfg.epsilon_start+p*(self.cfg.epsilon_end-self.cfg.epsilon_start)
    def act(self,obs):
        if self.rng.random()<self.epsilon: return int(self.rng.integers(3))
        with torch.inference_mode(): q=self.online(torch.from_numpy(obs).float().unsqueeze(0))[0].numpy()
        return int(self.rng.choice(np.flatnonzero(q==q.max())))
    def update(self,t):
        self.replay.add(t); self.steps+=1
        if self.replay.size<self.cfg.warmup or self.steps%self.cfg.update_every: return
        obs,actions,rewards,next_obs,done=self.replay.sample(self.cfg.batch_size,self.rng)
        predicted=self.online(obs).gather(1,actions[:,None]).squeeze(1)
        targets=double_targets(self.online,self.target,next_obs,rewards,done,self.cfg.gamma)
        loss=F.smooth_l1_loss(predicted,targets)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step(); self.updates+=1
        self.loss_sum+=float(loss.item()); self.loss_count+=1
        if self.updates%self.cfg.target_every==0: self.target.load_state_dict(self.online.state_dict())
    def snapshot(self): return {k:v.detach().cpu().numpy().copy() for k,v in self.online.state_dict().items()}
    def diagnostics(self):
        result=dict(epsilon=self.epsilon,replay_size=self.replay.size,optimizer_steps=self.updates,mean_td_loss=self.loss_sum/self.loss_count if self.loss_count else None)
        self.loss_sum=0.; self.loss_count=0
        return result
