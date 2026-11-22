"""Phase 5.5: dueling Double DQN with prioritized replay and n-step targets.

This is Rainbow-style, not the complete Rainbow algorithm (no C51/noisy nets).
The legacy DQN architecture remains available for old checkpoints.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tron_ai.observations import observation_shape
from .core import Transition
from .dqn import QNetwork, Replay

ARCHITECTURE = 'global-local-dueling-cnn-v2'


class DuelingNetwork(QNetwork):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(576,128), nn.ReLU())
        self.advantage = nn.Linear(128,3)
        self.value = nn.Linear(128,1)

    def forward(self, x):
        hidden = self.head(self.features(x))
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=1,keepdim=True)


def upgrade_legacy_weights(weights):
    """Function-preserving transformation from v1 Q-values to dueling v2 Q-values."""
    if 'advantage.weight' in weights:
        return {k: np.asarray(v).copy() for k,v in weights.items()}
    old = QNetwork().state_dict()
    if set(old) != set(weights) or any(tuple(old[k].shape)!=tuple(v.shape) for k,v in weights.items()):
        raise ValueError('DQN checkpoint is neither legacy v1 nor dueling v2')
    new = {k: np.asarray(v).copy() for k,v in weights.items() if not k.startswith('head.2.')}
    new['advantage.weight'] = np.asarray(weights['head.2.weight']).copy()
    new['advantage.bias'] = np.asarray(weights['head.2.bias']).copy()
    new['value.weight'] = np.asarray(weights['head.2.weight']).mean(axis=0,keepdims=True).copy()
    new['value.bias'] = np.asarray(weights['head.2.bias']).mean(axis=0,keepdims=True).copy()
    return new


def load_network(weights):
    with torch.random.fork_rng(devices=[]):
        model = DuelingNetwork()
    expected=model.state_dict()
    if set(expected)!=set(weights) or any(tuple(expected[k].shape)!=tuple(np.asarray(v).shape) or not np.isfinite(v).all() for k,v in weights.items()):
        raise ValueError('invalid enhanced DQN checkpoint')
    model.load_state_dict({k:torch.as_tensor(v).clone() for k,v in weights.items()})
    return model


class EnhancedDQNActor:
    def __init__(self, weights, board, seed):
        self.network=load_network(weights).eval().requires_grad_(False)
        self.shape=observation_shape(board)
        self.rng=np.random.default_rng(seed)

    def act(self,obs):
        if obs.shape != self.shape: raise ValueError('observation shape mismatch')
        with torch.inference_mode():
            q=self.network(torch.from_numpy(obs).float()[None])[0].numpy()
        return int(self.rng.choice(np.flatnonzero(q==q.max())))


def factory(weights,board,seed):
    return EnhancedDQNActor(weights,board,seed)


@dataclass(frozen=True)
class EnhancedDQNConfig:
    lr:float=0.00015
    gamma:float=.99
    capacity:int=32768
    batch_size:int=64
    warmup:int=512
    update_every:int=4
    target_every:int=500
    epsilon_start:float=.20
    epsilon_end:float=.03
    epsilon_decay:int=80000
    n_step:int=3
    alpha:float=.6
    beta_start:float=.4
    beta_steps:int=200000
    priority_eps:float=.001

    def __post_init__(self):
        vals=(self.lr,self.gamma,self.epsilon_start,self.epsilon_end,self.alpha,self.beta_start,self.priority_eps)
        if not all(math.isfinite(v) for v in vals) or not (self.lr>0 and 0<self.gamma<=1 and
          1<=self.batch_size<=self.warmup<=self.capacity and self.update_every>0 and self.target_every>0 and
          0<=self.epsilon_end<=self.epsilon_start<=1 and self.epsilon_decay>=1 and 1<=self.n_step<=16 and
          0<=self.alpha<=1 and 0<=self.beta_start<=1 and self.beta_steps>=1 and self.priority_eps>0):
            raise ValueError('invalid enhanced DQN config')


class PrioritizedReplay(Replay):
    """Packed observations and O(log N) segment-tree priority sampling."""
    def __init__(self, shape, capacity, alpha=.6):
        super().__init__(shape,capacity)
        self.alpha=alpha
        self.leaves=1<<(capacity-1).bit_length()
        self.tree=np.zeros(2*self.leaves,dtype=np.float64)
        self.discounts=np.zeros(capacity,dtype=np.float32)
        self.max_priority=1.

    def _set(self,i,priority):
        p=self.leaves+int(i); self.tree[p]=float(priority)**self.alpha
        while p>1:
            p//=2;self.tree[p]=self.tree[2*p]+self.tree[2*p+1]

    def add(self,t,discount=1.):
        i=self.pos
        super().add(t)
        self.discounts[i]=discount
        self._set(i,self.max_priority)

    def sample_prioritized(self,n,rng,beta):
        if not 1<=n<=self.size: raise ValueError('invalid sample size')
        total=self.tree[1]
        if total<=0: raise RuntimeError('priority tree is empty')
        idx=np.empty(n,dtype=np.int64)
        for j in range(n):
            mass=(j+rng.random())*total/n
            node=1
            while node<self.leaves:
                left=2*node
                if mass<self.tree[left]: node=left
                else: mass-=self.tree[left];node=left+1
            idx[j]=node-self.leaves
        probabilities=self.tree[self.leaves+idx]/total
        importance=np.power(np.maximum(probabilities*self.size,1e-12),-beta)
        importance/=importance.max()
        def unpack(storage):
            return torch.from_numpy(np.unpackbits(storage[idx],axis=1,count=self.bits).reshape(n,*self.shape)).float()
        batch=(unpack(self.obs),torch.from_numpy(self.actions[idx].copy()),
               torch.from_numpy(self.rewards[idx].copy()),unpack(self.next_obs),
               torch.from_numpy(self.done[idx].copy()),torch.from_numpy(self.discounts[idx].copy()))
        return batch,idx,torch.from_numpy(importance.astype(np.float32))

    def update_priorities(self,indices,errors,eps=.001):
        for i,error in zip(indices,errors):
            priority=float(abs(error)+eps)
            if not math.isfinite(priority): raise FloatingPointError('nonfinite priority')
            self.max_priority=max(self.max_priority,priority)
            self._set(int(i),priority)

    def export_state(self):
        return dict(shape=self.shape,capacity=self.capacity,alpha=self.alpha,pos=self.pos,size=self.size,
            obs=torch.from_numpy(self.obs[:self.size].copy()),next_obs=torch.from_numpy(self.next_obs[:self.size].copy()),
            actions=torch.from_numpy(self.actions[:self.size].copy()),rewards=torch.from_numpy(self.rewards[:self.size].copy()),
            done=torch.from_numpy(self.done[:self.size].copy()),discounts=torch.from_numpy(self.discounts[:self.size].copy()),
            tree=torch.from_numpy(self.tree.copy()),max_priority=self.max_priority)

    def import_state(self,state):
        if tuple(state['shape'])!=self.shape or state['capacity']!=self.capacity or state['alpha']!=self.alpha:
            raise ValueError('replay config mismatch')
        self.pos=int(state['pos']);self.size=int(state['size'])
        if not (0<=self.pos<self.capacity and 0<=self.size<=self.capacity): raise ValueError('invalid replay state')
        for name in ('obs','next_obs','actions','rewards','done','discounts'):
            x=state[name].numpy()
            if x.shape!=getattr(self,name)[:self.size].shape: raise ValueError('invalid replay array')
            getattr(self,name)[:self.size]=x
        if state['tree'].shape!=self.tree.shape: raise ValueError('invalid priority tree')
        self.tree[:]=state['tree'].numpy();self.max_priority=float(state['max_priority'])


class EnhancedDQNLearner:
    algorithm='dqn'
    def __init__(self,board,seed=2026,cfg=None,initial_weights=None):
        self.cfg=cfg or EnhancedDQNConfig()
        self.rng=np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online=DuelingNetwork();self.target=DuelingNetwork()
        if initial_weights is not None:
            self.online.load_state_dict(load_network(upgrade_legacy_weights(initial_weights)).state_dict())
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval().requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.cfg.lr)
        self.replay=PrioritizedReplay(observation_shape(board),self.cfg.capacity,self.cfg.alpha)
        self.n_buffer=deque()
        self.steps=self.updates=0;self.loss_sum=0.;self.loss_count=0

    @property
    def epsilon(self):
        fraction=min(1.,self.steps/self.cfg.epsilon_decay)
        return self.cfg.epsilon_start+fraction*(self.cfg.epsilon_end-self.cfg.epsilon_start)

    @property
    def beta(self):
        fraction=min(1.,self.steps/self.cfg.beta_steps)
        return self.cfg.beta_start+fraction*(1.-self.cfg.beta_start)

    def act(self,obs):
        if self.rng.random()<self.epsilon: return int(self.rng.integers(3))
        with torch.inference_mode():
            q=self.online(torch.from_numpy(obs).float()[None])[0].numpy()
        return int(self.rng.choice(np.flatnonzero(q==q.max())))

    def _push_nstep(self):
        seq=list(self.n_buffer)[:self.cfg.n_step]
        reward=0.; last=seq[-1];used=0
        for i,t in enumerate(seq):
            reward+=self.cfg.gamma**i*float(t.reward)
            last=t;used+=1
            if t.done: break
        first=self.n_buffer[0]
        self.replay.add(Transition(first.observation,first.action,reward,last.next_observation,last.done),
                        discount=self.cfg.gamma**used)
        self.n_buffer.popleft()

    def update(self,t):
        self.n_buffer.append(t);self.steps+=1
        if t.done:
            while self.n_buffer:self._push_nstep()
        elif len(self.n_buffer)>=self.cfg.n_step:
            self._push_nstep()
        if self.replay.size<self.cfg.warmup or self.steps%self.cfg.update_every:return
        (obs,actions,rewards,next_obs,done,discounts),idx,weights=self.replay.sample_prioritized(self.cfg.batch_size,self.rng,self.beta)
        q=self.online(obs).gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            next_actions=self.online(next_obs).argmax(1,keepdim=True)
            next_value=self.target(next_obs).gather(1,next_actions).squeeze(1)
            target=rewards+discounts*(~done).float()*next_value
        delta=q-target
        loss=(weights*F.smooth_l1_loss(q,target,reduction='none')).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite enhanced DQN loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        self.replay.update_priorities(idx,delta.detach().cpu().numpy(),self.cfg.priority_eps)
        self.loss_sum+=float(loss.detach());self.loss_count+=1
        if self.updates%self.cfg.target_every==0:self.target.load_state_dict(self.online.state_dict())

    def snapshot(self):
        return {k:v.detach().cpu().numpy().copy() for k,v in self.online.state_dict().items()}

    def diagnostics(self):
        result=dict(epsilon=self.epsilon,beta=self.beta,replay_size=self.replay.size,
                    optimizer_steps=self.updates,mean_td_loss=self.loss_sum/self.loss_count if self.loss_count else None)
        self.loss_sum=0.;self.loss_count=0
        return result

    def export_state(self):
        n=[]
        for t in self.n_buffer:
            n.append(dict(obs=torch.from_numpy(t.observation.copy()),action=int(t.action),
                          reward=float(t.reward),next_obs=torch.from_numpy(t.next_observation.copy()),done=bool(t.done)))
        return dict(config=asdict(self.cfg),online=self.online.state_dict(),target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(),replay=self.replay.export_state(),pending_nstep=n,
                    rng=self.rng.bit_generator.state,steps=self.steps,updates=self.updates,
                    loss_sum=self.loss_sum,loss_count=self.loss_count)

    def import_state(self,state):
        if state['config']!=asdict(self.cfg):raise ValueError('enhanced DQN config does not match resume file')
        self.online.load_state_dict(state['online']);self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer']);self.replay.import_state(state['replay'])
        self.n_buffer=deque(Transition(item['obs'].numpy().copy(),int(item['action']),float(item['reward']),
                                       item['next_obs'].numpy().copy(),bool(item['done'])) for item in state['pending_nstep'])
        self.rng.bit_generator.state=state['rng'];self.steps=int(state['steps']);self.updates=int(state['updates'])
        self.loss_sum=float(state['loss_sum']);self.loss_count=int(state['loss_count'])
