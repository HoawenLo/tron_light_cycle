"""Phase 5.6: stable, weights-only DQN fine-tuning using ordinary game experience.

A frozen copy of the starting Q-network protects previously useful predictions on
real transitions from a small archive. This is Q-value anchoring, not imitation
learning: no scripted action labels or teacher dataset are used.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F
from tron_ai.observations import observation_shape
from .advanced_dqn import (EnhancedDQNConfig, EnhancedDQNLearner,
                           PrioritizedReplay, DuelingNetwork)
from .core import Transition


@dataclass(frozen=True)
class StableDQNConfig(EnhancedDQNConfig):
    lr: float = 0.00003
    epsilon_start: float = .06
    epsilon_end: float = .02
    epsilon_decay: int = 120000
    alpha: float = .4
    target_every: int = 750
    archive_capacity: int = 8192
    archive_fraction: float = .25
    anchor_coef: float = .10

    def __post_init__(self):
        super().__post_init__()
        if not (self.archive_capacity >= self.batch_size and
                math.isfinite(self.archive_fraction) and 0 <= self.archive_fraction < 1 and
                math.isfinite(self.anchor_coef) and self.anchor_coef >= 0):
            raise ValueError('invalid stable DQN configuration')


class StableDQNLearner(EnhancedDQNLearner):
    """Dueling Double DQN with protected old-policy experience and Q anchoring."""
    def __init__(self, board, seed=2026, cfg=None, initial_weights=None):
        if initial_weights is None:
            raise ValueError('stable fine-tuning needs a pretrained DQN checkpoint')
        super().__init__(board, seed, cfg or StableDQNConfig(), initial_weights)
        self.protected = PrioritizedReplay(observation_shape(board),
                                           self.cfg.archive_capacity, self.cfg.alpha)
        self.protected_buffer = deque()
        self.anchor = DuelingNetwork()
        self.anchor.load_state_dict(self.online.state_dict())
        self.anchor.eval().requires_grad_(False)
        self.protected_episodes = 0
        self.protected_samples = 0
        self.anchor_penalty_sum = 0.
        self.anchor_penalty_count = 0
        self.q_abs_sum = 0.
        self.td_abs_sum = 0.

    def _push_protected(self):
        seq=list(self.protected_buffer)[:self.cfg.n_step]
        reward=0.;last=seq[-1];used=0
        for i,t in enumerate(seq):
            reward += self.cfg.gamma**i * float(t.reward)
            last=t;used+=1
            if t.done:break
        first=self.protected_buffer[0]
        self.protected.add(Transition(first.observation,first.action,reward,
                                      last.next_observation,last.done),
                           discount=self.cfg.gamma**used)
        self.protected_buffer.popleft()

    def record_protected(self, transition):
        """Record ordinary transitions from the old agent; never optimise here."""
        self.protected_buffer.append(transition)
        if transition.done:
            while self.protected_buffer:self._push_protected()
            self.protected_episodes += 1
        elif len(self.protected_buffer)>=self.cfg.n_step:
            self._push_protected()

    def update(self, transition):
        self.n_buffer.append(transition)
        self.steps += 1
        if transition.done:
            while self.n_buffer:self._push_nstep()
        elif len(self.n_buffer)>=self.cfg.n_step:
            self._push_nstep()
        if self.replay.size < self.cfg.warmup or self.steps % self.cfg.update_every:
            return

        keep=min(int(round(self.cfg.batch_size*self.cfg.archive_fraction)),
                 self.protected.size)
        count=self.cfg.batch_size-keep
        (obs,actions,rewards,next_obs,done,discounts),fresh_idx,fresh_weights = (
            self.replay.sample_prioritized(count,self.rng,self.beta))
        if keep:
            (old_obs,old_actions,old_rewards,old_next,old_done,old_discount),old_idx,old_weights = (
                self.protected.sample_prioritized(keep,self.rng,self.beta))
            obs=torch.cat((obs,old_obs))
            actions=torch.cat((actions,old_actions))
            rewards=torch.cat((rewards,old_rewards))
            next_obs=torch.cat((next_obs,old_next))
            done=torch.cat((done,old_done))
            discounts=torch.cat((discounts,old_discount))
            sample_weights=torch.cat((fresh_weights,old_weights))
        else:
            sample_weights=fresh_weights

        q_all=self.online(obs)
        q=q_all.gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            next_actions=self.online(next_obs).argmax(1,keepdim=True)
            next_value=self.target(next_obs).gather(1,next_actions).squeeze(1)
            target=rewards+discounts*(~done).float()*next_value
        delta=q-target
        td_loss=(sample_weights*F.smooth_l1_loss(q,target,reduction='none')).mean()
        if keep:
            with torch.no_grad():
                anchor_q=self.anchor(obs[-keep:])
            anchor_penalty=F.smooth_l1_loss(q_all[-keep:],anchor_q)
        else:
            anchor_penalty=q.new_zeros(())
        loss=td_loss+self.cfg.anchor_coef*anchor_penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite stable DQN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step()
        self.updates+=1
        errors=delta.detach().cpu().numpy()
        self.replay.update_priorities(fresh_idx,errors[:count],self.cfg.priority_eps)
        if keep:
            self.protected.update_priorities(old_idx,errors[count:],self.cfg.priority_eps)
        self.loss_sum+=float(loss.detach());self.loss_count+=1
        self.anchor_penalty_sum+=float(anchor_penalty.detach());self.anchor_penalty_count+=1
        self.q_abs_sum+=float(q.detach().abs().mean())
        self.td_abs_sum+=float(delta.detach().abs().mean())
        self.protected_samples+=keep
        if self.updates%self.cfg.target_every==0:
            self.target.load_state_dict(self.online.state_dict())

    def diagnostics(self):
        count=self.anchor_penalty_count
        metrics=dict(protected_replay_size=self.protected.size,
                     protected_episodes=self.protected_episodes,
                     cumulative_protected_samples=self.protected_samples,
                     anchor_penalty=self.anchor_penalty_sum/count if count else None,
                     mean_abs_q=self.q_abs_sum/count if count else None,
                     mean_abs_td_error=self.td_abs_sum/count if count else None)
        metrics.update(super().diagnostics())
        self.anchor_penalty_sum=0.;self.anchor_penalty_count=0
        self.q_abs_sum=0.;self.td_abs_sum=0.
        return metrics

    def export_state(self):
        state=super().export_state()
        state.update(protected_archive=self.protected.export_state(),
                     anchor_network=self.anchor.state_dict(),
                     protected_episodes=self.protected_episodes,
                     protected_samples=self.protected_samples,
                     protected_pending=[dict(obs=torch.from_numpy(t.observation.copy()),
                         action=int(t.action),reward=float(t.reward),
                         next_obs=torch.from_numpy(t.next_observation.copy()),done=bool(t.done))
                         for t in self.protected_buffer],
                     anchor_penalty_sum=self.anchor_penalty_sum,
                     anchor_penalty_count=self.anchor_penalty_count,
                     q_abs_sum=self.q_abs_sum,td_abs_sum=self.td_abs_sum)
        return state

    def import_state(self,state):
        super().import_state(state)
        self.protected.import_state(state['protected_archive'])
        self.anchor.load_state_dict(state['anchor_network'])
        self.anchor.eval().requires_grad_(False)
        self.protected_episodes=int(state['protected_episodes'])
        self.protected_samples=int(state['protected_samples'])
        self.protected_buffer=deque(Transition(t['obs'].numpy().copy(),int(t['action']),
            float(t['reward']),t['next_obs'].numpy().copy(),bool(t['done']))
            for t in state['protected_pending'])
        self.anchor_penalty_sum=float(state['anchor_penalty_sum'])
        self.anchor_penalty_count=int(state['anchor_penalty_count'])
        self.q_abs_sum=float(state['q_abs_sum'])
        self.td_abs_sum=float(state['td_abs_sum'])
