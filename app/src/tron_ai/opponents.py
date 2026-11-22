"""Scripted reference policies. All act on the same pre-move state."""
from __future__ import annotations

from collections import deque
import random
from typing import Protocol

from tron_ai.engine import Action, Position, TronConfig, TronState, VECTORS, next_position, turn


class Policy(Protocol):
    def choose_action(self, state: TronState, player: int, config: TronConfig) -> Action: ...


def in_bounds(pos: Position, config: TronConfig) -> bool:
    return 0 <= pos[0] < config.width and 0 <= pos[1] < config.height


def legal_moves(state: TronState, player: int, config: TronConfig) -> dict[Action, Position]:
    """Immediate wall/trail-safe moves; a simultaneous head-on remains possible."""
    occupied = state.occupied
    moves = {}
    for action in Action:
        target = next_position(state.heads[player], turn(state.directions[player], action))
        if in_bounds(target, config) and target not in occupied:
            moves[action] = target
    return moves


def distances(start: Position, blocked: frozenset[Position] | set[Position],
              config: TronConfig) -> dict[Position, int]:
    """Flood fill from a free cell, returning shortest path lengths."""
    if not in_bounds(start, config) or start in blocked:
        return {}
    result = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in VECTORS:
            pos = (x + dx, y + dy)
            if pos not in result and pos not in blocked and in_bounds(pos, config):
                result[pos] = result[(x, y)] + 1
                queue.append(pos)
    return result


OPPONENTS = {
    'random': 'Random actions',
    'safe': 'Random safe',
    'space': 'Flood-fill survival',
    'territory': 'Territory control',
}


class ScriptedPolicy:
    def __init__(self, kind: str = 'space', *, seed: int | None = None):
        if kind not in OPPONENTS:
            raise ValueError(f'unknown opponent: {kind}')
        self.kind = kind
        self.rng = random.Random(seed)

    def choose_action(self, state: TronState, player: int, config: TronConfig) -> Action:
        if state.done:
            raise ValueError('cannot play a finished state')
        if player not in (0, 1):
            raise ValueError('player must be 0 or 1')
        if self.kind == 'random':
            return self.rng.choice(list(Action))
        moves = legal_moves(state, player, config)
        if not moves:
            # All moves crash; the simultaneous engine still decides win/loss/draw.
            return self.rng.choice(list(Action))
        if self.kind == 'safe':
            return self.rng.choice(list(moves))
        blocked = state.occupied
        other_moves = legal_moves(state, 1 - player, config)
        scores = {}
        for action, target in moves.items():
            own = distances(target, blocked, config)
            escapes = sum(d == 1 for d in own.values())
            if self.kind == 'space':
                scores[action] = (len(own), escapes, target not in other_moves.values())
                continue
            # Worst-case one-turn lookahead over all safe opponent replies.
            # Distance-based territory, ties neutral. No observed human action.
            replies = []
            for enemy in other_moves.values():
                if target == enemy:
                    replies.append((0, 0, 0))  # simultaneous draw
                    continue
                a = distances(target, blocked | {enemy}, config)
                b = distances(enemy, blocked | {target}, config)
                own_area = sum(p not in b or d < b[p] for p, d in a.items())
                enemy_area = sum(p not in a or d < a[p] for p, d in b.items())
                # Avoid a known next-turn trap even if area score is attractive.
                own_exit = any(d == 1 for d in a.values())
                enemy_exit = any(d == 1 for d in b.values())
                outcome = int(own_exit) - int(enemy_exit)
                replies.append((outcome, own_area - enemy_area, len(a)))
            scores[action] = min(replies) if replies else (1, len(own), escapes)
        best = max(scores.values())
        return self.rng.choice([action for action, score in scores.items() if score == best])
