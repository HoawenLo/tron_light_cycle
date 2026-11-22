"""Fixed, paired reference evaluations; independent of training randomness."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import random
from tron_ai.engine import TronConfig
from tron_ai.opponents import OPPONENTS, ScriptedPolicy
from .core import Opening, ActorFactory, play_episode
import numpy as np

VALIDATION_SEED = 8675309
HOLDOUT_SEED = 314159265

@dataclass(frozen=True)
class Case:
    opponent: str
    pair: int
    opening: Opening


def schedule(pairs: int, seed: int = VALIDATION_SEED) -> list[Case]:
    if pairs < 1:
        raise ValueError('pairs must be positive')
    rng = random.Random(seed)
    cases = []
    for i in range(pairs):
        board_seed, actor_seed = rng.getrandbits(63), rng.getrandbits(63)
        seeds = {op: rng.getrandbits(63) for op in OPPONENTS}
        for op in OPPONENTS:
            for side in (0, 1):
                cases.append(Case(op, i, Opening(board_seed, ('horizontal','vertical')[i%2],
                                               False, side, actor_seed, seeds[op])))
    return cases


def summarise(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('no games')
    def metrics(group):
        n = len(group)
        w = sum(r['reward']==1 for r in group)
        d = sum(r['reward']==0 for r in group)
        l = n-w-d
        counts = [sum(r['actions'][a] for r in group) for a in range(3)]
        total = sum(counts)
        return dict(games=n, wins=w, draws=d, losses=l, win_rate=w/n,
                    draw_rate=d/n, loss_rate=l/n, score=(w+.5*d)/n,
                    mean_reward=(w-l)/n, mean_steps=sum(r['steps'] for r in group)/n,
                    action_fractions=[c/total if total else 0 for c in counts])
    return {**metrics(rows), 'by_opponent': {op: metrics([r for r in rows if r['opponent']==op])
            for op in OPPONENTS if any(r['opponent']==op for r in rows)}}


def evaluate(factory: ActorFactory, weights: dict[str,np.ndarray], config: TronConfig,
             cases: list[Case]) -> tuple[dict,list[dict]]:
    rows = []
    for case in cases:
        o = case.opening
        # Fresh actor and opponent every match, never call the training learner.
        actor = factory({k:v.copy() for k,v in weights.items()}, config, o.actor_seed)
        game = play_episode(config,o,actor,ScriptedPolicy(case.opponent,seed=o.opponent_seed))
        rows.append({'opponent':case.opponent,'pair':case.pair,**asdict(o),**asdict(game)})
    return summarise(rows), rows
