"""Shared full-board encoding for DQN, PPO, and Neuroevolution.

Format: a contiguous, channel-first np.uint8 array (13, S, S), where
S=max(board_width, board_height).  The rectangular board is centred within
an S x S canvas. Areas outside the board are explicitly 1 in channel 12.

The entire canvas is rotated counter-clockwise by own heading, so every
observer sees its own cycle facing NORTH. The own/opponent semantic channels
are swapped by player. Directions are encoded *after rotation*.

channel  0: own visited trail, INCLUDING current head
channel  1: opponent visited trail, INCLUDING current head
channel  2: own head
channel  3: opponent head
channels 4..7: own head with direction N/E/S/W (own is always N)
channels 8..11: opponent head with direction N/E/S/W, relative to own heading
channel 12: padded out-of-board cells (wall); physical board edges are the
            border between wall and in-bounds cells, or square edge.

Using uint8 avoids unnecessary replay-buffer memory; networks can cast to
float32 when preparing batches. All three algorithms must use this encoder.
"""

from __future__ import annotations

import numpy as np

from tron_ai.engine import TronConfig, TronState

CHANNEL_COUNT = 13


def observation_shape(config: TronConfig) -> tuple[int, int, int]:
    size = max(config.width, config.height)
    return CHANNEL_COUNT, size, size


def encode_observation(state: TronState, player: int,
                       config: TronConfig) -> np.ndarray:
    """Encode one player's full information; no hidden or privileged fields."""
    if player not in (0, 1) or isinstance(player, bool):
        raise ValueError("player must be 0 or 1")
    size = max(config.width, config.height)
    x_offset, y_offset = (size - config.width) // 2, (size - config.height) // 2
    canvas = np.zeros((CHANNEL_COUNT, size, size), dtype=np.uint8)
    canvas[12] = 1
    canvas[12, y_offset:y_offset + config.height,
           x_offset:x_offset + config.width] = 0
    other = 1 - player
    for channel, owner in ((0, player), (1, other)):
        for x, y in state.trails[owner]:
            canvas[channel, y + y_offset, x + x_offset] = 1
    for channel, owner in ((2, player), (3, other)):
        x, y = state.heads[owner]
        canvas[channel, y + y_offset, x + x_offset] = 1

    k = int(state.directions[player])
    rotated = np.rot90(canvas, k=k, axes=(1, 2)).copy()
    # On the rotated canvas, the viewer's own direction is always north.
    for channel, owner in ((4, player), (8, other)):
        rel_direction = (int(state.directions[owner]) - k) % 4
        y, x = np.argwhere(rotated[2 if owner == player else 3] == 1)[0]
        rotated[channel + rel_direction, y, x] = 1
    return rotated
