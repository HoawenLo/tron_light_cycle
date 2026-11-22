"""Lightweight NumPy multi-scale CNN used by reward-only neuroevolution.

No learned features, masks or labels are provided beyond the same 13-channel
player-relative board observation supplied to the other algorithms.  The
7x7 crop is *derived* from that observation; it is not extra information.

CNN architecture v3: 13 -> 8 -> 12 spatial channels, stride 2 on each
3x3 convolution. The first scale is adaptively pooled to 5x5; the second
scale is retained at its native resolution to preserve spatial location.
A 7x7 local crop is independently embedded before a 32-wide fusion layer.
Weights are evolved directly, never trained with gradients.
"""
from __future__ import annotations
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tron_ai.observations import observation_shape

CHANNELS = 13
FIRST = 8
SECOND = 12
LOCAL = 7
LOCAL_HIDDEN = 16
FUSION = 32
FIRST_POOL = 5


def _dimension(board):
    _, height, width = observation_shape(board)
    if height != width:
        raise ValueError('CNN requires the square, player-relative observation')
    return height


def shapes(board):
    side = _dimension(board)
    second_side = math.ceil(math.ceil(side / 2) / 2)
    fused = FIRST * FIRST_POOL**2 + SECOND * second_side**2 + LOCAL_HIDDEN
    return {
        'conv1_w': (FIRST, CHANNELS, 3, 3),
        'conv1_b': (FIRST,),
        'conv2_w': (SECOND, FIRST, 3, 3),
        'conv2_b': (SECOND,),
        'local_w': (CHANNELS * LOCAL * LOCAL, LOCAL_HIDDEN),
        'local_b': (LOCAL_HIDDEN,),
        'fusion_w': (fused, FUSION),
        'fusion_b': (FUSION,),
        'output_w': (FUSION, 3),
        'output_b': (3,),
    }


def initialise(board, rng):
    params = {key: np.zeros(dims, dtype=np.float32) for key, dims in shapes(board).items()}
    for key in ('conv1_w', 'conv2_w', 'local_w', 'fusion_w', 'output_w'):
        arr = params[key]
        fan_in = math.prod(arr.shape[1:]) if key.startswith('conv') else arr.shape[0]
        params[key] = rng.normal(0, math.sqrt(2 / fan_in), arr.shape).astype(np.float32)
    return params


def conv_stride2(inputs, kernel, bias):
    """Valid same-padded convolution with stride 2, C,H,W -> K,ceil(H/2),ceil(W/2).

    One im2col/matmul operation per convolution, with small 3x3 kernels.
    """
    if inputs.ndim != 3 or kernel.ndim != 4 or kernel.shape[1] != inputs.shape[0] or kernel.shape[2:] != (3, 3):
        raise ValueError('incompatible convolution shapes')
    padded = np.pad(inputs, ((0, 0), (1, 1), (1, 1)), mode='constant')
    windows = sliding_window_view(padded, (3, 3), axis=(1, 2))[:, ::2, ::2]
    c, h, w, _, _ = windows.shape
    patches = windows.transpose(1, 2, 0, 3, 4).reshape(h*w, -1)
    out = patches @ kernel.reshape(kernel.shape[0], -1).T
    return np.maximum(out.reshape(h, w, -1).transpose(2, 0, 1) + bias[:, None, None], 0)


def adaptive_pool(inputs, bins=FIRST_POOL):
    """Pool a channel-first spatial map without discarding locations outside a bin.

    This is the *early* convolutional scale; the later convolutional scale
    stays unpooled for the final fusion layer.
    """
    channels, height, width = inputs.shape
    # The standard 30x24 board produces a 15x15 first-scale feature map.
    # Exact 5x5 block means can be computed with one contiguous NumPy reduction.
    if height % bins == 0 and width % bins == 0:
        return inputs.reshape(channels, bins, height // bins, bins, width // bins).mean(
            axis=(2, 4), dtype=np.float32)
    out = np.zeros((channels, bins, bins), dtype=np.float32)
    for iy in range(bins):
        y0 = (iy * height) // bins
        y1 = max(y0 + 1, math.ceil((iy + 1) * height / bins))
        for ix in range(bins):
            x0 = (ix * width) // bins
            x1 = max(x0 + 1, math.ceil((ix + 1) * width / bins))
            out[:, iy, ix] = inputs[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return out


def head_patch(obs, side=LOCAL):
    if side < 1 or side % 2 == 0:
        raise ValueError('local patch size must be positive and odd')
    _, height, width = obs.shape
    y, x = divmod(int(obs[2].argmax()), width)
    radius = side // 2
    patch = np.zeros((CHANNELS, side, side), dtype=np.float32)
    patch[12] = 1  # out-of-board cells are marked as boundaries
    y0, y1 = max(0, y-radius), min(height, y+radius+1)
    x0, x1 = max(0, x-radius), min(width, x+radius+1)
    patch[:, y0-y+radius:y1-y+radius, x0-x+radius:x1-x+radius] = obs[:, y0:y1, x0:x1]
    return patch.reshape(-1)


def forward(obs, weights):
    x = np.asarray(obs, dtype=np.float32)
    early = conv_stride2(x, weights['conv1_w'], weights['conv1_b'])
    deep = conv_stride2(early, weights['conv2_w'], weights['conv2_b'])
    local = np.maximum(head_patch(obs) @ weights['local_w'] + weights['local_b'], 0)
    features = np.concatenate((adaptive_pool(early).reshape(-1), deep.reshape(-1), local))
    hidden = np.maximum(features @ weights['fusion_w'] + weights['fusion_b'], 0)
    return hidden @ weights['output_w'] + weights['output_b']
