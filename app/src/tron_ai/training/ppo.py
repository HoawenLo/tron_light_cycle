"""CPU PPO with categorical policy, GAE and strictly on-policy rollouts."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tron_ai.observations import observation_shape
from .dqn import QNetwork

ARCHITECTURE = 'global-local-actor-critic-v1'

@dataclass(frozen=True)
class PPOConfig:
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 512
    batch_size: int = 64
    epochs: int = 4
    clip: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    def __post_init__(self):
        nums = (self.lr,self.gamma,self.gae_lambda,self.clip,self.entropy_coef,self.value_coef,self.max_grad_norm)
        if not (all(math.isfinite(n) for n in nums) and self.lr>0 and 0<=self.gamma<=1 and 0<=self.gae_lambda<=1 and 2<=self.batch_size<=self.rollout_steps and self.epochs>=1 and 0<self.clip<1 and self.entropy_coef>=0 and self.value_coef>=0 and self.max_grad_norm>0):
            raise ValueError('invalid PPO configuration')
        if self.rollout_steps % self.batch_size:
            raise ValueError('rollout_steps must be divisible by batch_size')

class ActorCritic(QNetwork):
    """Reuse identical global/local feature extraction; output logits and V(s)."""
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(576,128),nn.ReLU())
        self.policy=nn.Linear(128,3)
        self.value=nn.Linear(128,1)
        # Near-uniform initial policy, avoiding arbitrary action preference.
        nn.init.orthogonal_(self.policy.weight,gain=.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight,gain=1.)
        nn.init.zeros_(self.value.bias)
    def forward(self,x):
        h=self.head(self.features(x))
        return self.policy(h), self.value(h).squeeze(-1)


def compute_gae(rewards, values, dones, bootstrap, gamma, lam):
    """Terminal masks prevent propagation between independent matches."""
    rewards=np.asarray(rewards,dtype=np.float32)
    values=np.asarray(values,dtype=np.float32)
    dones=np.asarray(dones,dtype=bool)
    if rewards.ndim!=1 or rewards.shape!=values.shape or rewards.shape!=dones.shape or len(rewards)==0:
        raise ValueError('invalid GAE arrays')
    advantages=np.empty_like(rewards); carry=0.
    for i in range(len(rewards)-1,-1,-1):
        continuation=0. if dones[i] else 1.
        nxt=bootstrap if i==len(rewards)-1 else values[i+1]
        delta=rewards[i]+gamma*continuation*nxt-values[i]
        carry=delta+gamma*lam*continuation*carry
        advantages[i]=carry
    return advantages, advantages+values


def clipped_policy_loss(new_logprob, old_logprob, advantages, clip):
    ratio=(new_logprob-old_logprob).exp()
    return -torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages).mean()


def load_network(weights):
    with torch.random.fork_rng(devices=[]): net=ActorCritic()
    expected=net.state_dict()
    if set(expected)!=set(weights) or any(expected[k].shape!=weights[k].shape or not np.isfinite(weights[k]).all() for k in expected):
        raise ValueError('invalid PPO weights')
    net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
    return net

class PPOActor:
    """Frozen evaluation/self-play actor, sampling from the learned policy."""
    def __init__(self,weights,board,seed):
        self.network=load_network(weights).eval().requires_grad_(False)
        self.rng=np.random.default_rng(seed)
        self.shape=observation_shape(board)
    def act(self,obs):
        if obs.shape!=self.shape: raise ValueError('observation shape mismatch')
        with torch.inference_mode(): logits,_=self.network(torch.from_numpy(obs).float()[None])
        p=logits[0].double().softmax(-1).numpy()
        return int(self.rng.choice(3,p=p))

def factory(weights,board,seed): return PPOActor(weights,board,seed)

class PPOLearner:
    algorithm='ppo'
    def __init__(self,board,seed=2026,cfg=None,initial_weights=None):
        self.cfg=cfg or PPOConfig(); self.rng=np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.network=ActorCritic()
        if initial_weights is not None:
            self.network.load_state_dict(load_network(initial_weights).state_dict())
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=self.cfg.lr,eps=1e-5)
        self.shape=observation_shape(board); self.bits=math.prod(self.shape)
        n=self.cfg.rollout_steps
        self.obs=np.empty((n,(self.bits+7)//8),np.uint8)
        self.actions=np.empty(n,np.int64); self.logprobs=np.empty(n,np.float32)
        self.values=np.empty(n,np.float32); self.rewards=np.empty(n,np.float32); self.dones=np.empty(n,bool)
        self.size=self.steps=self.updates=self.rollouts=0
        self.pending=None; self.stats=[]; self.action_counts=np.zeros(3,dtype=np.int64)
        self.entropy_sum=0.; self.entropy_count=0
    def act(self,obs):
        if self.pending is not None: raise RuntimeError('act must be followed by update')
        if obs.shape!=self.shape: raise ValueError('observation shape mismatch')
        with torch.inference_mode():
            logits,v=self.network(torch.from_numpy(obs).float()[None]); lp=logits[0].double().log_softmax(-1)
            p=lp.exp().numpy(); a=int(self.rng.choice(3,p=p)); logp=float(lp[a]); value=float(v[0])
            self.entropy_sum+=float(-(lp.exp()*lp).sum()); self.entropy_count+=1
        self.pending=(np.packbits(obs.reshape(-1)),a,logp,value)
        return a
    def update(self,t):
        if self.pending is None or self.pending[1]!=int(t.action): raise RuntimeError('transition does not match sampled action')
        packed,a,lp,v=self.pending; self.pending=None
        if not np.array_equal(packed,np.packbits(t.observation.reshape(-1))): raise RuntimeError('transition observation mismatch')
        i=self.size
        self.obs[i]=packed; self.actions[i]=a; self.logprobs[i]=lp; self.values[i]=v
        self.rewards[i]=t.reward; self.dones[i]=t.done
        self.size+=1; self.steps+=1; self.action_counts[a]+=1
        if self.size==self.cfg.rollout_steps:
            with torch.inference_mode():
                bootstrap=0. if t.done else float(self.network(torch.from_numpy(t.next_observation).float()[None])[1][0])
            self._optimise(bootstrap)
    def _optimise(self,bootstrap):
        n=self.size; cfg=self.cfg
        adv,ret=compute_gae(self.rewards[:n],self.values[:n],self.dones[:n],bootstrap,cfg.gamma,cfg.gae_lambda)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(cfg.epochs):
            order=self.rng.permutation(n)
            for start in range(0,n,cfg.batch_size):
                idx=order[start:start+cfg.batch_size]
                obs=torch.from_numpy(np.unpackbits(self.obs[idx],axis=1,count=self.bits).reshape(len(idx),*self.shape)).float()
                logits,value=self.network(obs); dist=Categorical(logits=logits)
                lp=dist.log_prob(torch.from_numpy(self.actions[idx])); old=torch.from_numpy(self.logprobs[idx]); a=torch.from_numpy(adv[idx]); r=torch.from_numpy(ret[idx])
                policy_loss=clipped_policy_loss(lp,old,a,cfg.clip)
                value_loss=(value-r).square().mean(); entropy=dist.entropy().mean()
                loss=policy_loss+cfg.value_coef*value_loss-cfg.entropy_coef*entropy
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite PPO loss')
                with torch.no_grad():
                    logratio=lp-old; ratio=logratio.exp()
                    kl=((ratio-1)-logratio).mean(); frac=((ratio-1).abs()>cfg.clip).float().mean()
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),cfg.max_grad_norm,error_if_nonfinite=True)
                self.optimizer.step(); self.updates+=1
                self.stats.append([float(x.detach()) for x in (policy_loss,value_loss,entropy,kl,frac)])
        self.rollouts+=1; self.size=0
    def snapshot(self): return {k:v.detach().cpu().numpy().copy() for k,v in self.network.state_dict().items()}
    def diagnostics(self):
        keys=('policy_loss','value_loss','update_entropy','approx_kl','clip_fraction')
        metrics=dict(zip(keys,np.mean(self.stats,axis=0).tolist() if self.stats else [None]*len(keys)))
        total=int(self.action_counts.sum())
        metrics.update(optimizer_steps=self.updates,completed_rollouts=self.rollouts,pending_rollout_steps=self.size,
                       policy_entropy=self.entropy_sum/self.entropy_count if self.entropy_count else None,
                       training_action_fractions=(self.action_counts/total).tolist() if total else None)
        self.stats.clear(); self.action_counts.fill(0); self.entropy_sum=0.; self.entropy_count=0
        return metrics
