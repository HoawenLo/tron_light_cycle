"""Phase 5.5 PPO: clipped value losses, KL stop and step-based LR annealing.

The actor-critic architecture and inference checkpoint format are unchanged.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math
import numpy as np
import torch
from torch.distributions import Categorical
from .ppo import PPOLearner,PPOConfig,compute_gae,clipped_policy_loss


@dataclass(frozen=True)
class EnhancedPPOConfig(PPOConfig):
    lr:float=.00015
    entropy_coef:float=.012
    target_kl:float=.025
    value_clip:float=.2
    lr_end:float=.000025
    anneal_steps:int=250000

    def __post_init__(self):
        super().__post_init__()
        if not (math.isfinite(self.target_kl) and self.target_kl>0 and
                math.isfinite(self.value_clip) and self.value_clip>0 and
                math.isfinite(self.lr_end) and 0<self.lr_end<=self.lr and self.anneal_steps>=1):
            raise ValueError('invalid enhanced PPO configuration')


class EnhancedPPOLearner(PPOLearner):
    def __init__(self,board,seed=2026,cfg=None,initial_weights=None):
        super().__init__(board,seed,cfg or EnhancedPPOConfig(),initial_weights)
        self.early_stops=0

    @property
    def current_lr(self):
        factor=min(1.,self.steps/self.cfg.anneal_steps)
        return self.cfg.lr+(self.cfg.lr_end-self.cfg.lr)*factor

    def _optimise(self,bootstrap):
        n=self.size;cfg=self.cfg
        adv,ret=compute_gae(self.rewards[:n],self.values[:n],self.dones[:n],bootstrap,cfg.gamma,cfg.gae_lambda)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        for group in self.optimizer.param_groups:group['lr']=self.current_lr
        stopped=False
        for _ in range(cfg.epochs):
            if stopped:break
            order=self.rng.permutation(n)
            for start in range(0,n,cfg.batch_size):
                idx=order[start:start+cfg.batch_size]
                obs=torch.from_numpy(np.unpackbits(self.obs[idx],axis=1,count=self.bits).reshape(len(idx),*self.shape)).float()
                logits,value=self.network(obs); dist=Categorical(logits=logits)
                lp=dist.log_prob(torch.from_numpy(self.actions[idx]));old=torch.from_numpy(self.logprobs[idx])
                a=torch.from_numpy(adv[idx]);r=torch.from_numpy(ret[idx]);old_value=torch.from_numpy(self.values[idx])
                policy_loss=clipped_policy_loss(lp,old,a,cfg.clip)
                clipped_value=old_value+(value-old_value).clamp(-cfg.value_clip,cfg.value_clip)
                v_loss=torch.maximum((value-r).square(),(clipped_value-r).square()).mean()
                entropy=dist.entropy().mean()
                loss=policy_loss+cfg.value_coef*v_loss-cfg.entropy_coef*entropy
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite enhanced PPO loss')
                with torch.no_grad():
                    logratio=lp-old; ratio=logratio.exp()
                    kl=((ratio-1)-logratio).mean()
                    frac=((ratio-1).abs()>cfg.clip).float().mean()
                if float(kl)>1.5*cfg.target_kl:
                    self.early_stops+=1;stopped=True;break
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),cfg.max_grad_norm,error_if_nonfinite=True)
                self.optimizer.step();self.updates+=1
                self.stats.append([float(x.detach()) for x in (policy_loss,v_loss,entropy,kl,frac)])
        self.rollouts+=1;self.size=0

    def diagnostics(self):
        d=super().diagnostics();d['learning_rate']=self.current_lr;d['kl_early_stops']=self.early_stops
        return d

    def export_state(self):
        n=self.size
        return dict(config=asdict(self.cfg),network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),
          rng=self.rng.bit_generator.state,size=n,steps=self.steps,updates=self.updates,rollouts=self.rollouts,
          obs=torch.from_numpy(self.obs[:n].copy()),actions=torch.from_numpy(self.actions[:n].copy()),
          logprobs=torch.from_numpy(self.logprobs[:n].copy()),values=torch.from_numpy(self.values[:n].copy()),
          rewards=torch.from_numpy(self.rewards[:n].copy()),dones=torch.from_numpy(self.dones[:n].copy()),
          pending=None if self.pending is None else dict(packed=torch.from_numpy(self.pending[0].copy()),
                         action=self.pending[1],logp=self.pending[2],value=self.pending[3]),
          stats=self.stats,action_counts=torch.from_numpy(self.action_counts.copy()),
          entropy_sum=self.entropy_sum,entropy_count=self.entropy_count,early_stops=self.early_stops)

    def import_state(self,state):
        if state['config']!=asdict(self.cfg):raise ValueError('enhanced PPO config does not match resume file')
        self.network.load_state_dict(state['network']);self.optimizer.load_state_dict(state['optimizer'])
        n=int(state['size'])
        if not 0<=n<=self.cfg.rollout_steps:raise ValueError('invalid PPO buffer size')
        for key in ('obs','actions','logprobs','values','rewards','dones'):
            target=getattr(self,key);source=state[key].numpy()
            if source.shape!=target[:n].shape: raise ValueError('invalid PPO rollout array')
            target[:n]=source
        self.size=n;self.steps=int(state['steps']);self.updates=int(state['updates']);self.rollouts=int(state['rollouts'])
        self.rng.bit_generator.state=state['rng']
        pending=state['pending']
        self.pending=None if pending is None else (pending['packed'].numpy().copy(),int(pending['action']),
                                                   float(pending['logp']),float(pending['value']))
        self.stats=list(state['stats']);self.action_counts[:]=state['action_counts'].numpy()
        self.entropy_sum=float(state['entropy_sum']);self.entropy_count=int(state['entropy_count'])
        self.early_stops=int(state['early_stops'])
