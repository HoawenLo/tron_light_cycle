"""Deterministic, renderer-free, simultaneous two-player Tron Light Cycles.

All state transitions are pure via ``advance``.  The state contains every
previously occupied cell, INCLUDING each cycle's current head.  Every turn:

1. Decode both relative turns using their PREVIOUS headings.
2. Compute BOTH intended destinations using the PREVIOUS state.
3. Resolve BOTH collisions against the PREVIOUS occupied cells.
4. Commit safe moves together, or terminate the game.

This order deliberately prevents a first-mover advantage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, IntEnum
from numbers import Integral
from typing import Iterable

Position = tuple[int, int]  # (column x, row y); origin is top-left


class Action(IntEnum):
    """Discrete RL actions; these numeric IDs remain stable for all agents."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Direction(IntEnum):
    """Compass directions, clockwise, with y increasing down the board."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Collision(str, Enum):
    WALL = "wall"
    OWN_TRAIL = "own_trail"
    OPPONENT_TRAIL = "opponent_trail"
    HEAD_ON = "head_on"
    HEAD_SWAP = "head_swap"


class EndReason(str, Enum):
    PLAYER_CRASH = "player_crash"
    DOUBLE_CRASH = "double_crash"
    HEAD_ON = "head_on"
    HEAD_SWAP = "head_swap"
    STEP_LIMIT = "step_limit"


VECTORS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True, slots=True)
class TronConfig:
    width: int = 30
    height: int = 24
    max_steps: int | None = None

    def __post_init__(self) -> None:
        for name, val, smallest in (("width", self.width, 8), ("height", self.height, 6)):
            if isinstance(val, bool) or not isinstance(val, Integral) or val < smallest:
                raise ValueError(f"{name} must be an integer >= {smallest}")
        limit = self.width * self.height if self.max_steps is None else self.max_steps
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
            raise ValueError("max_steps must be a positive integer")
        object.__setattr__(self, "max_steps", int(limit))


@dataclass(frozen=True, slots=True)
class TronState:
    """Immutable logical state. Each trail includes the corresponding head."""

    heads: tuple[Position, Position]
    directions: tuple[Direction, Direction]
    trails: tuple[frozenset[Position], frozenset[Position]]
    step_count: int = 0
    done: bool = False
    winner: int | None = None  # None is ongoing or a draw
    end_reason: EndReason | None = None

    @property
    def occupied(self) -> frozenset[Position]:
        return self.trails[0] | self.trails[1]

    @property
    def is_draw(self) -> bool:
        return self.done and self.winner is None


@dataclass(frozen=True, slots=True)
class StepResult:
    previous_state: TronState
    state: TronState
    actions: tuple[Action, Action]
    attempted_heads: tuple[Position, Position]
    collisions: tuple[Collision | None, Collision | None]
    rewards: tuple[int, int]

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def winner(self) -> int | None:
        return self.state.winner

    @property
    def end_reason(self) -> EndReason | None:
        return self.state.end_reason


def _position_is_valid(pos: Position, config: TronConfig) -> bool:
    return (isinstance(pos, tuple) and len(pos) == 2
            and all(isinstance(v, Integral) and not isinstance(v, bool) for v in pos)
            and 0 <= pos[0] < config.width and 0 <= pos[1] < config.height)


def validate_state(state: TronState, config: TronConfig) -> None:
    """Reject corrupt custom/replayed states before advancing or observing."""
    if not isinstance(state, TronState):
        raise TypeError("state must be a TronState")
    if len(state.heads) != 2 or len(state.directions) != 2 or len(state.trails) != 2:
        raise ValueError("state must contain exactly two players")
    if any(not _position_is_valid(pos, config) for pos in state.heads):
        raise ValueError("both heads must be inside the board")
    if state.heads[0] == state.heads[1]:
        raise ValueError("heads may not overlap")
    if any(not isinstance(d, Direction) for d in state.directions):
        raise ValueError("both directions must be Direction values")
    if any(not isinstance(trail, frozenset) for trail in state.trails):
        raise ValueError("trails must be frozensets for immutable states")
    for player in (0, 1):
        if state.heads[player] not in state.trails[player]:
            raise ValueError("each trail must include its player's head")
        if any(not _position_is_valid(pos, config) for pos in state.trails[player]):
            raise ValueError("every trail cell must be inside the board")
    if not state.trails[0].isdisjoint(state.trails[1]):
        raise ValueError("trails cannot overlap")
    if (isinstance(state.step_count, bool) or not isinstance(state.step_count, Integral)
            or not 0 <= state.step_count <= config.max_steps):
        raise ValueError("invalid step_count")
    if not isinstance(state.done, bool):
        raise ValueError("done must be bool")
    if state.done:
        if not isinstance(state.end_reason, EndReason):
            raise ValueError("terminal states require an EndReason")
        if state.winner not in (0, 1, None):
            raise ValueError("winner must be 0, 1, or None")
        if state.end_reason is EndReason.STEP_LIMIT:
            if state.winner is not None or state.step_count != config.max_steps:
                raise ValueError("step-limit terminations must be draws at max_steps")
        elif state.end_reason is EndReason.PLAYER_CRASH:
            if state.winner not in (0, 1):
                raise ValueError("a single crash must have a winner")
        elif state.winner is not None:
            raise ValueError("a mutual crash must be a draw")
    elif state.winner is not None or state.end_reason is not None:
        raise ValueError("nonterminal state cannot have a winner or end reason")
    elif state.step_count >= config.max_steps:
        raise ValueError("a state at the step limit must be terminal")


def _parse_actions(actions: Iterable[int | Action]) -> tuple[Action, Action]:
    try:
        pair = tuple(actions)
    except TypeError as exc:
        raise ValueError("provide exactly two actions") from exc
    if len(pair) != 2:
        raise ValueError("provide exactly two actions")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in pair):
        raise ValueError("actions must be integer IDs 0=left, 1=straight, 2=right")
    try:
        return Action(pair[0]), Action(pair[1])
    except ValueError as exc:
        raise ValueError("actions must be 0=left, 1=straight, 2=right") from exc


def turn(direction: Direction, action: Action) -> Direction:
    """Relative left/straight/right; a reverse turn is not an action."""
    return Direction((int(direction) + int(action) - 1) % 4)


def next_position(head: Position, direction: Direction) -> Position:
    dx, dy = VECTORS[int(direction)]
    return head[0] + dx, head[1] + dy


def advance(state: TronState, actions: Iterable[int | Action],
            config: TronConfig) -> StepResult:
    """Compute an atomic simultaneous turn without mutating ``state``.

    One-player wall/trail crashes cause a win for the other player; two crashes
    are draws.  Both entering the same FREE cell is a head-on draw.  Exchanging
    head positions is also a draw, because both heads are already occupied.
    Failed moves do not add invalid coordinates to trails.  ``attempted_heads``
    exposes crash destinations to a future renderer.
    """
    if state.done:
        raise RuntimeError("cannot advance a finished game; call reset()")
    pair = _parse_actions(actions)
    # Both actions and next positions are calculated from the identical old state.
    headings = (turn(state.directions[0], pair[0]),
                turn(state.directions[1], pair[1]))
    targets = (next_position(state.heads[0], headings[0]),
               next_position(state.heads[1], headings[1]))

    collisions: list[Collision | None] = [None, None]
    for player in (0, 1):
        target = targets[player]
        if not _position_is_valid(target, config):
            collisions[player] = Collision.WALL
        elif target in state.trails[player]:
            collisions[player] = Collision.OWN_TRAIL
        elif target in state.trails[1 - player]:
            collisions[player] = Collision.OPPONENT_TRAIL

    # A move into the same vacant cell: neither player is considered first.
    if targets[0] == targets[1] and collisions == [None, None]:
        collisions = [Collision.HEAD_ON, Collision.HEAD_ON]
    # Adjacent heads exchanging cells are *not* allowed to pass through each other.
    elif targets == (state.heads[1], state.heads[0]):
        collisions = [Collision.HEAD_SWAP, Collision.HEAD_SWAP]

    hit_0, hit_1 = (collision is not None for collision in collisions)
    if hit_0 and hit_1:
        done, winner = True, None
        if collisions == [Collision.HEAD_ON, Collision.HEAD_ON]:
            reason = EndReason.HEAD_ON
        elif collisions == [Collision.HEAD_SWAP, Collision.HEAD_SWAP]:
            reason = EndReason.HEAD_SWAP
        else:
            reason = EndReason.DOUBLE_CRASH
    elif hit_0 or hit_1:
        done, winner, reason = True, (1 if hit_0 else 0), EndReason.PLAYER_CRASH
    elif state.step_count + 1 >= config.max_steps:
        done, winner, reason = True, None, EndReason.STEP_LIMIT
    else:
        done, winner, reason = False, None, None

    new_heads = tuple(state.heads[p] if collisions[p] else targets[p] for p in (0, 1))
    new_trails = tuple(
        state.trails[p] if collisions[p] else state.trails[p] | {targets[p]}
        for p in (0, 1)
    )
    updated = TronState(
        heads=new_heads,  # type: ignore[arg-type]
        directions=headings,
        trails=new_trails,  # type: ignore[arg-type]
        step_count=state.step_count + 1,
        done=done,
        winner=winner,
        end_reason=reason,
    )
    if not done or winner is None:
        rewards = (0, 0)
    else:
        rewards = (1, -1) if winner == 0 else (-1, 1)
    return StepResult(state, updated, pair, targets, tuple(collisions), rewards)


class TronEnv:
    """Small stateful wrapper around the pure engine for games and RL runners.

    Both policies must select actions *before* calling ``step((a0, a1))``.
    Any reads of ``observe(0)`` and ``observe(1)`` before that call use the
    same frozen state, regardless of which policy is asked first.
    """

    def __init__(self, config: TronConfig | None = None, *, seed: int | None = None):
        self.config = config if config is not None else TronConfig()
        self._rng = random.Random(seed)
        self.state = self.reset()

    def reset(self, *, seed: int | None = None, mode: str = "fixed",
              swap_sides: bool | None = None, axis: str | None = None) -> TronState:
        """Start a new game with symmetrical positions, both facing inward.

        ``fixed`` uses fixed centre-row/centre-column opening distances;
        ``random_mirrored`` varies mirrored spawn positions and starting
        side assignments. ``axis`` can be "horizontal" or "vertical".
        When omitted, fixed games are horizontal and random-mirrored games
        randomly choose the axis, allowing all four initial headings across
        training episodes. Use explicit axes and side-swaps to construct
        controlled/balanced evaluation sets. Passing ``seed`` resets the
        independent local RNG for exact replay.
        """
        if seed is not None:
            self._rng.seed(seed)
        if mode not in ("fixed", "random_mirrored"):
            raise ValueError("mode must be 'fixed' or 'random_mirrored'")
        if axis is not None and axis not in ("horizontal", "vertical"):
            raise ValueError("axis must be 'horizontal', 'vertical', or None")
        if swap_sides is not None and not isinstance(swap_sides, bool):
            raise ValueError("swap_sides must be True, False or None")
        if axis is None:
            axis = ("horizontal" if mode == "fixed" else
                    self._rng.choice(("horizontal", "vertical")))
        if mode == "fixed":
            if axis == "horizontal":
                near = self.config.width // 4
                middle = self.config.height // 2
            else:
                near = self.config.height // 4
                middle = self.config.width // 2
            swapped = False if swap_sides is None else swap_sides
        else:
            if axis == "horizontal":
                near = self._rng.randint(1, (self.config.width - 3) // 2)
                middle = self._rng.randint(1, self.config.height - 2)
            else:
                near = self._rng.randint(1, (self.config.height - 3) // 2)
                middle = self._rng.randint(1, self.config.width - 2)
            swapped = bool(self._rng.getrandbits(1)) if swap_sides is None else swap_sides
        if axis == "horizontal":
            first = (near, middle)    # left side
            second = (self.config.width - 1 - near, middle)  # right side
            headings = (Direction.EAST, Direction.WEST)
        else:
            first = (middle, near)   # top side
            second = (middle, self.config.height - 1 - near)  # bottom side
            headings = (Direction.SOUTH, Direction.NORTH)
        heads = (second, first) if swapped else (first, second)
        directions = headings[::-1] if swapped else headings
        self.state = TronState(
            heads=heads,
            directions=directions,
            trails=(frozenset({heads[0]}), frozenset({heads[1]})),
        )
        return self.state

    def load_state(self, state: TronState) -> TronState:
        """Load a fully validated state for replay or controlled test setups."""
        validate_state(state, self.config)
        self.state = state
        return state

    def step(self, actions: Iterable[int | Action]) -> StepResult:
        result = advance(self.state, actions, self.config)
        self.state = result.state
        return result

    def observe(self, player: int):
        """Get the same-format, player-relative observation for either policy."""
        from tron_ai.observations import encode_observation
        return encode_observation(self.state, player, self.config)
