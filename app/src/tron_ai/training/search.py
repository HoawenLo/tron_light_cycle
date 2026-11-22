"""CPU-only antithetic Evolution Strategies and Augmented Random Search.

Both methods optimise the same terminal reward (+1 / 0 / -1) using mirrored
Gaussian perturbations and shared paired games.  No backpropagation, policy
masks, game-specific safety heuristics, or additional rewards are used.

The learner emits ordinary 'evolution' weight checkpoints to retain compatibility
with Pygame, self-play, benchmark and existing train/evaluate tooling.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from tron_ai.opponents import ScriptedPolicy
from .core import ObservationPolicy, play_episode
from .evolution import (ARCHITECTURES, architecture_of, copy_weights, factory,
                        initialise, make_pair_schedule, parameter_count)


@dataclass(frozen=True)
class SearchConfig:
    optimizer: str = 'es'
    architecture: str = 'linear'
    directions: int = 12
    top_directions: int = 6
    fitness_pairs: int = 8
    noise_std: float = .025
    step_size: float = .01
    max_update_l2: float = 1.0
    survival_safe_fraction: float = .5

    def __post_init__(self):
        if (self.optimizer not in ('es', 'ars') or self.architecture not in ARCHITECTURES
                or self.directions < 1 or not 1 <= self.top_directions <= self.directions
                or self.fitness_pairs < 1 or not math.isfinite(self.noise_std)
                or self.noise_std <= 0 or not math.isfinite(self.step_size)
                or self.step_size <= 0 or not math.isfinite(self.max_update_l2)
                or self.max_update_l2 <= 0 or not math.isfinite(self.survival_safe_fraction)
                or not 0 < self.survival_safe_fraction < 1):
            raise ValueError('invalid antithetic-search configuration')


def mirrored_weights(base, direction, sigma: float, sign: int):
    """Return a fresh positive or negative perturbation without aliasing base."""
    if sign not in (-1, 1) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('invalid perturbation')
    return {k: (v + np.float32(sign * sigma) * direction[k]).astype(np.float32)
            for k, v in base.items()}


def antithetic_update(optimizer: str, directions: list[dict[str, np.ndarray]],
                      scores: np.ndarray, *, step_size: float, noise_std: float,
                      top_directions: int, max_update_l2: float):
    """Pure, deterministic weight update, with no update on zero fitness signal.

    ES: raw mirrored finite difference / (2 * number_of_directions * sigma).
    ARS: top-k mirrored directions by max(+fitness,-fitness), normalised by
    std of their 2k returns (Mania et al.).  No distributional scaling if all
    fitness values are equal; ARS naturally takes a zero step in this case.
    Both paths cap global L2 update length for numerical stability.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(directions)
    if (optimizer not in ('es', 'ars') or n == 0 or scores.shape != (n, 2)
            or not np.isfinite(scores).all() or not 1 <= top_directions <= n
            or not math.isfinite(noise_std) or noise_std <= 0
            or not math.isfinite(step_size) or step_size <= 0
            or not math.isfinite(max_update_l2) or max_update_l2 <= 0):
        raise ValueError('invalid mirrored fitness update')
    keys = set(directions[0])
    if any(set(d) != keys or any(d[k].shape != directions[0][k].shape for k in keys)
           for d in directions):
        raise ValueError('noise directions have incompatible shapes')
    if optimizer == 'ars':
        # Stable sorting preserves deterministic selection for tied fitness.
        selected = np.argsort(-scores.max(axis=1), kind='stable')[:top_directions]
        scale = float(np.std(scores[selected].ravel()))
        coeff = step_size / (len(selected) * scale) if scale > 1e-12 else 0.
    else:
        selected = np.arange(n)
        scale = 2.0 * n * noise_std
        coeff = step_size / scale
    diff = scores[:, 0] - scores[:, 1]
    # Accumulate in float32 for CPU cost and consistent dtype with policies.
    updates = {k: np.zeros_like(directions[0][k], dtype=np.float32) for k in keys}
    if coeff > 0:
        for index in selected:
            multiplier = np.float32(coeff * diff[index])
            if multiplier:
                for k in keys:
                    updates[k] += multiplier * directions[index][k]
    norm = math.sqrt(sum(float(np.sum(np.square(u, dtype=np.float64))) for u in updates.values()))
    if norm > max_update_l2:
        factor = np.float32(max_update_l2 / norm)
        updates = {k: v * factor for k, v in updates.items()}
    return updates, dict(update_l2=min(norm, max_update_l2), unclipped_update_l2=norm,
                         directions_used=len(selected), fitness_spread=float(np.std(scores)),
                         signal_fraction=float(np.mean(diff != 0)),
                         reward_scale=scale if optimizer == 'ars' else None)


class SearchLearner:
    """Mirrored-policy, common-random-numbers, fixed-budget search learner."""
    algorithm = 'evolution'

    def __init__(self, board, seed=2026, cfg=None, initial_weights=None):
        self.cfg = cfg or SearchConfig()
        self.rng = np.random.default_rng(seed)
        if initial_weights is None:
            self.weights = initialise(board, self.rng, self.cfg.architecture)
        else:
            if architecture_of(initial_weights, board) != self.cfg.architecture:
                raise ValueError('warm-start architecture mismatch')
            self.weights = copy_weights(initial_weights)
        self.generation = 0
        self.fitness = None
        self.last_generation_record = None
        self.parameter_count = parameter_count(board, self.cfg.architecture)
        self.last_update = None

    @property
    def batch_episodes(self):
        return 2 * self.cfg.directions * 2 * self.cfg.fitness_pairs

    def snapshot(self):
        return copy_weights(self.weights)

    def diagnostics(self):
        return dict(optimizer=self.cfg.optimizer, architecture=self.cfg.architecture,
                    generation=self.generation, directions=self.cfg.directions,
                    parameters=self.parameter_count,
                    best_fitness=None if self.fitness is None else float(np.max(self.fitness)),
                    mean_fitness=None if self.fitness is None else float(np.mean(self.fitness)),
                    worst_fitness=None if self.fitness is None else float(np.min(self.fitness)),
                    fitness_std=None if self.fitness is None else float(np.std(self.fitness)),
                    **(self.last_update or {}))

    def train_batch(self, board, pool, open_rng, *, progress=0.):
        pairs = make_pair_schedule(board, pool, open_rng, self.cfg.fitness_pairs,
                                   self.generation, self.cfg.survival_safe_fraction)
        # Identical opening/opponent conditions for +eps and -eps in *every* direction.
        directions = [
            {k: self.rng.standard_normal(size=v.shape).astype(np.float32)
             for k, v in self.weights.items()}
            for _ in range(self.cfg.directions)
        ]
        rows = []
        fitness = np.empty((self.cfg.directions, 2), dtype=np.float64)
        opponent_rewards = {name: [] for _, name, _, _ in pairs}
        for i, direction in enumerate(directions):
            for j, sign in enumerate((1, -1)):
                trial = mirrored_weights(self.weights, direction, self.cfg.noise_std, sign)
                rewards = []
                for pair_index, name, historical, openings in pairs:
                    for opening in openings:
                        if historical is None:
                            opponent = ScriptedPolicy(name, seed=opening.opponent_seed)
                        else:
                            opponent = ObservationPolicy(historical.factory(
                                copy_weights(historical.weights), board, opening.opponent_seed))
                        result = play_episode(board, opening,
                                              factory(trial, board, opening.actor_seed), opponent)
                        rewards.append(result.reward)
                        opponent_rewards[name].append(result.reward)
                        rows.append(dict(generation=self.generation + 1, candidate=i,
                                         perturbation_sign=sign, pair_index=pair_index,
                                         opponent=name, opening=asdict(opening), **asdict(result)))
                fitness[i, j] = np.mean(rewards)
        updates, stats = antithetic_update(
            self.cfg.optimizer, directions, fitness, step_size=self.cfg.step_size,
            noise_std=self.cfg.noise_std, top_directions=self.cfg.top_directions,
            max_update_l2=self.cfg.max_update_l2)
        self.weights = {k: (v + updates[k]).astype(np.float32)
                        for k, v in self.weights.items()}
        self.generation += 1
        self.fitness = fitness
        self.last_update = stats
        self.last_generation_record = dict(
            generation=self.generation, stage=pool.stage, optimizer=self.cfg.optimizer,
            architecture=self.cfg.architecture, parameters=self.parameter_count,
            population=2 * self.cfg.directions, fitness_pairs=self.cfg.fitness_pairs,
            total_games=len(rows), opponent_pairs=dict(Counter(name for _, name, _, _ in pairs)),
            best_fitness=float(fitness.max()), mean_fitness=float(fitness.mean()),
            worst_fitness=float(fitness.min()), fitness_std=float(fitness.std()),
            opponent_mean_reward={name: float(np.mean(values))
                                  for name, values in opponent_rewards.items()},
            best_direction=int(np.argmax(fitness.max(axis=1))),
            next_offspring_mutation_sigma=None, noise_std=self.cfg.noise_std,
            **stats)
        return rows
