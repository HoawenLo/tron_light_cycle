"""Human game controller, independent of Pygame for testing and reuse."""
from __future__ import annotations

from enum import Enum
from tron_ai.engine import Action, Direction, TronConfig, TronEnv, StepResult
from tron_ai.opponents import OPPONENTS, Policy, ScriptedPolicy


class Phase(str, Enum):
    READY = 'ready'
    COUNTDOWN = 'countdown'
    PLAYING = 'playing'
    PAUSED = 'paused'
    FINISHED = 'finished'


class GameSession:
    def __init__(self, *, config: TronConfig | None = None, seed: int = 2026,
                 opponent: str = 'space', speed: int = 8, extra_opponents=None):
        self.env = TronEnv(config, seed=seed)
        self.seed = seed
        self.round_number = 0
        self.speed = max(3, min(16, speed))
        self.extra_opponents = dict(extra_opponents or {})
        self.opponents = {**OPPONENTS, **{key: label for key, (label, _) in self.extra_opponents.items()}}
        self.scores = {key: [0, 0, 0] for key in self.opponents}
        self.opponent = opponent
        self.policy: Policy
        self.phase = Phase.READY
        self.pending = Action.STRAIGHT
        self.last_result: StepResult | None = None
        self.countdown = 0.0
        self.elapsed = 0.0
        self.prepare()

    def prepare(self) -> None:
        # Alternate starting sides, fixed horizontal opening for human matches.
        self.env.reset(swap_sides=bool(self.round_number % 2))
        seed = self.seed + self.round_number
        self.policy = (self.extra_opponents[self.opponent][1](seed) if self.opponent in self.extra_opponents
                       else ScriptedPolicy(self.opponent, seed=seed))
        self.phase = Phase.READY
        self.pending = Action.STRAIGHT
        self.last_result = None
        self.countdown = self.elapsed = 0.0

    def select_opponent(self, kind: str) -> bool:
        if kind not in self.opponents:
            raise ValueError(f'unknown opponent: {kind}')
        if self.phase not in (Phase.READY, Phase.FINISHED):
            return False
        self.opponent = kind
        self.prepare()
        return True

    def start_or_pause(self) -> None:
        if self.phase == Phase.FINISHED:
            self.prepare()
        if self.phase == Phase.READY:
            self.phase = Phase.COUNTDOWN
            self.countdown = 3.0
            self.elapsed = 0.0
        elif self.phase == Phase.PLAYING:
            self.phase = Phase.PAUSED
            self.elapsed = 0.0
        elif self.phase == Phase.PAUSED:
            self.phase = Phase.COUNTDOWN
            self.countdown = 3.0
            self.elapsed = 0.0

    def steer(self, heading: Direction) -> bool:
        if self.phase not in (Phase.PLAYING, Phase.COUNTDOWN):
            return False
        diff = (int(heading) - int(self.env.state.directions[0])) % 4
        # Only one non-straight turn may be buffered per simulation tick.
        if diff not in (1, 3) or self.pending != Action.STRAIGHT:
            return False
        self.pending = Action.RIGHT if diff == 1 else Action.LEFT
        return True

    def step(self) -> StepResult | None:
        if self.phase != Phase.PLAYING:
            return None
        state = self.env.state
        enemy = self.policy.choose_action(state, 1, self.env.config)
        result = self.env.step((self.pending, enemy))
        self.last_result = result
        self.pending = Action.STRAIGHT
        if result.done:
            self.phase = Phase.FINISHED
            index = 2 if result.winner is None else result.winner
            self.scores[self.opponent][index] += 1
            self.round_number += 1
        return result

    def update(self, seconds: float) -> None:
        if self.phase == Phase.COUNTDOWN:
            self.countdown = max(0.0, self.countdown - seconds)
            if self.countdown <= 1e-9:
                self.phase = Phase.PLAYING
                self.elapsed = 0.0
            return
        if self.phase != Phase.PLAYING:
            return
        # Limit to one tick per rendered frame to avoid sudden multi-tick jumps.
        self.elapsed += min(max(0.0, seconds), 0.25)
        period = 1 / self.speed
        if self.elapsed >= period:
            self.elapsed = min(self.elapsed - period, period)
            self.step()

    def change_speed(self, delta: int) -> None:
        self.speed = max(3, min(16, self.speed + delta))
        self.elapsed = 0.0
