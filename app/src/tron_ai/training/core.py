"""Common observation-only learner API and atomic single-learner matches."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable
import random
import numpy as np
from tron_ai.engine import Action, TronConfig, TronEnv, TronState
from tron_ai.opponents import Policy

@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: Action
    reward: int
    next_observation: np.ndarray
    done: bool

class Actor(Protocol):
    def act(self, observation: np.ndarray) -> int: ...

class Learner(Protocol):
    algorithm: str
    def act(self, observation: np.ndarray) -> int: ...
    def update(self, transition: Transition) -> None: ...
    def snapshot(self) -> dict[str, np.ndarray]: ...

# Factory receives immutable-by-convention copied weights, config and action seed.
ActorFactory = Callable[[dict[str, np.ndarray], TronConfig, int], Actor]

class ObservationPolicy:
    """Adapt observation-only neural actors to the existing scripted/Pygame API."""
    def __init__(self, actor: Actor):
        self.actor = actor
    def choose_action(self, state: TronState, player: int, config: TronConfig) -> Action:
        from tron_ai.observations import encode_observation
        return Action(self.actor.act(encode_observation(state, player, config)))

@dataclass(frozen=True)
class Opening:
    board_seed: int
    axis: str
    swap: bool
    learner_side: int
    actor_seed: int
    opponent_seed: int

@dataclass(frozen=True)
class Episode:
    reward: int
    steps: int
    end_reason: str
    actions: tuple[int, int, int]


def play_episode(config: TronConfig, opening: Opening, actor: Actor,
                 opponent: Policy, *, update: Callable[[Transition], None] | None = None) -> Episode:
    """The actor and opponent both see the old state. No legal-action mask or shaping."""
    env = TronEnv(config)
    env.reset(seed=opening.board_seed, mode='random_mirrored', axis=opening.axis,
              swap_sides=opening.swap)
    side = opening.learner_side
    if side not in (0, 1):
        raise ValueError('learner_side must be 0 or 1')
    obs = env.observe(side)
    counts = [0, 0, 0]
    while not env.state.done:
        raw = actor.act(obs)
        # Rely on the engine's strict parsing for action IDs (including invalid types).
        other = opponent.choose_action(env.state, 1-side, config)
        actions = (raw, other) if side == 0 else (other, raw)
        result = env.step(actions)
        action = result.actions[side]
        counts[int(action)] += 1
        next_obs = env.observe(side)
        if update is not None:
            update(Transition(obs, action, result.rewards[side], next_obs, result.done))
        obs = next_obs
    return Episode(result.rewards[side], env.state.step_count, env.state.end_reason.value, tuple(counts))


def training_opening(rng: random.Random, episode_index: int) -> Opening:
    # Separate RNG stream from evaluation and opponent selection.
    return Opening(rng.getrandbits(63), rng.choice(('horizontal','vertical')),
                   False, episode_index % 2, rng.getrandbits(63), rng.getrandbits(63))
