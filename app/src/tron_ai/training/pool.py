"""Versioned opponent sampling; historical actors are isolated from learner updates."""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
from tron_ai.engine import TronConfig
from tron_ai.opponents import ScriptedPolicy, Policy
from .core import ActorFactory, ObservationPolicy

# Conditional scripted weights sum to 1. Self-play uses 40% history when available.
STAGES = {
    'survival': {'random': .5, 'safe': .5},
    'space': {'random': .1, 'safe': .45, 'space': .45},
    'competitive': {'random': .05, 'safe': .20, 'space': .45, 'territory': .30},
    'selfplay': {'random': .05, 'safe': .20, 'space': .45, 'territory': .30},
    'league': {'random': .10, 'safe': .30, 'space': .55, 'territory': .05},
    # Both focused stages use scripted opponents only (no historical self-play).
    'territory_focus': {'random': .05, 'safe': .10, 'space': .25, 'territory': .60},
    'strong_balance': {'safe': .10, 'space': .45, 'territory': .45},
}

@dataclass
class Frozen:
    name: str
    weights: dict[str,np.ndarray]
    score: float
    factory: ActorFactory

class OpponentPool:
    def __init__(self, stage: str, *, seed: int, capacity: int = 12):
        if stage not in STAGES or capacity < 2:
            raise ValueError('invalid stage or capacity < 2')
        self.stage, self.capacity = stage, capacity
        self.rng = random.Random(seed)
        self.progress = 0.0
        self.history: list[Frozen] = []

    def add(self, name: str, weights: dict[str,np.ndarray], score: float,
            factory: ActorFactory) -> None:
        if any(f.name == name for f in self.history):
            raise ValueError('duplicate snapshot')
        self.history.append(Frozen(name,{k:v.copy() for k,v in weights.items()},score,factory))
        if len(self.history)>self.capacity:
            best = max(self.history,key=lambda f:f.score)
            # Keep the best and recent historical snapshots, evict oldest non-best.
            self.history.remove(next(f for f in self.history if f is not best))

    def export_state(self):
        import torch
        return dict(stage=self.stage,capacity=self.capacity,progress=self.progress,
                    rng_state=self.rng.getstate(),history=[dict(name=f.name,score=f.score,
                    weights={k:torch.from_numpy(v.copy()) for k,v in f.weights.items()}) for f in self.history])

    def import_state(self,state,factory):
        if state['capacity']!=self.capacity or state['stage'] not in STAGES:
            raise ValueError('opponent pool configuration mismatch')
        self.stage=state['stage'];self.progress=float(state['progress'])
        self.rng.setstate(state['rng_state'])
        self.history=[Frozen(item['name'],{k:v.numpy().copy() for k,v in item['weights'].items()},
                             float(item['score']),factory) for item in state['history']]
        if len(self.history)>self.capacity:raise ValueError('opponent history overflow')

    def sample(self, config: TronConfig, seed: int) -> tuple[str,Policy]:
        league=self.stage=='league'
        historical=(self.stage=='selfplay' and self.history and self.rng.random()<.4) or (
                    league and len(self.history)>1 and self.progress>.1 and self.rng.random()<.20)
        if historical:
            mode = self.rng.choice(('best','latest','uniform'))
            f = (max(self.history,key=lambda f:f.score) if mode=='best' else
                 self.history[-1] if mode=='latest' else self.rng.choice(self.history))
            return f.name, ObservationPolicy(f.factory({k:v.copy() for k,v in f.weights.items()},config,seed))
        if league:
            # Low-pressure warm-up. The scheduled rise is intentionally bounded:
            # the league never allocates over 15% of scripted games to Territory.
            p=max(0.,min(1.,self.progress))
            w={'random': .10-.05*p,'safe': .30-.05*p,'space': .55-.05*p,'territory': .05+.15*p}
        else:
            w = STAGES[self.stage]
        name = self.rng.choices(list(w),weights=list(w.values()),k=1)[0]
        return name, ScriptedPolicy(name, seed=seed)
