"""Small REINFORCE plumbing check, NOT the planned PPO/DQN/evolution agents."""
from __future__ import annotations
import numpy as np
from tron_ai.engine import TronConfig
from .core import Transition

ALGORITHM = 'reference'

def features(obs: np.ndarray) -> np.ndarray:
    # Input is the shared full-board array; this toy internally extracts local features.
    y,x = np.argwhere(obs[2])[0]
    blocked = np.any(obs[[0,1,12]],axis=0)
    h,w = blocked.shape
    free = [float(0<=yy<h and 0<=xx<w and not blocked[yy,xx])
            for yy,xx in ((y,x-1),(y-1,x),(y,x+1))]
    return np.array([1.,*free],dtype=np.float64)

class ReferenceActor:
    def __init__(self, weights: dict[str,np.ndarray], seed: int):
        self.weights = np.asarray(weights['policy'],dtype=np.float64).copy()
        if self.weights.shape != (3,4) or not np.isfinite(self.weights).all():
            raise ValueError('invalid reference weights')
        self.rng = np.random.default_rng(seed)
    def probabilities(self, obs):
        z = self.weights @ features(obs)
        p = np.exp(z-z.max())
        return p/p.sum()
    def act(self, observation):
        return int(self.rng.choice(3,p=self.probabilities(observation)))

class ReferenceLearner(ReferenceActor):
    algorithm = ALGORITHM
    def __init__(self, seed: int, lr: float=.05):
        super().__init__({'policy':np.zeros((3,4))},seed)
        self.lr = lr
        self.gradients = []
    def update(self, transition: Transition):
        p = self.probabilities(transition.observation)
        p[int(transition.action)] -= 1
        self.gradients.append(-np.outer(p,features(transition.observation)))
        if transition.done:
            # Undiscounted sparse terminal reward; averaged episode policy gradient.
            self.weights += self.lr * transition.reward * np.mean(self.gradients,axis=0)
            self.gradients.clear()
    def snapshot(self):
        return {'policy':self.weights.copy()}

def factory(weights: dict[str,np.ndarray], config: TronConfig, seed: int):
    return ReferenceActor(weights,seed)
