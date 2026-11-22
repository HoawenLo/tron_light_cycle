"""Reward-only neuroevolution with balanced, paired candidate evaluations.

The original 192k-parameter architecture remains supported for checkpoint inference
and explicit legacy training. The default compact network still *receives* the
same 13-channel full-board observation, but summarises its global spatial
information with deterministic pooling and retains an exact local head patch.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
import math
import numpy as np
from tron_ai.observations import observation_shape
from tron_ai.opponents import ScriptedPolicy
from .core import Opening, ObservationPolicy, play_episode
from .pool import STAGES
from . import evolution_cnn

ARCHITECTURES = {
    'compact': 'pooled-full-board-and-local-mlp-v2',
    'legacy': 'full-board-local-mlp-v1',
    'cnn': 'multiscale-conv-local7-v3',
    'linear': 'linear-full-board-v4',
}
ARCHITECTURE = ARCHITECTURES['compact']
HIDDEN = 16
CHANNELS = 13
POOL = 6
PATCH = 5
LOCAL_FEATURES = CHANNELS * PATCH * PATCH
GLOBAL_FEATURES = CHANNELS * POOL * POOL * 2  # mean and max: no new game information


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 16
    elites: int = 4
    fitness_pairs: int = 16
    mutation_sigma: float = .02
    mutation_rate: float = .01
    architecture: str = 'compact'
    fitness_sampling: str = 'balanced'
    survival_safe_fraction: float = .5
    mutation_schedule: str = 'fixed'
    mutation_sigma_end: float = .005

    def __post_init__(self):
        if (self.population < 2 or self.elites < 1 or self.elites >= self.population
                or self.fitness_pairs < 1 or not math.isfinite(self.mutation_sigma)
                or self.mutation_sigma <= 0 or not math.isfinite(self.mutation_rate)
                or not 0 < self.mutation_rate <= 1
                or self.architecture not in ARCHITECTURES
                or self.fitness_sampling not in ('balanced', 'random')
                or not math.isfinite(self.survival_safe_fraction)
                or not 0 < self.survival_safe_fraction < 1
                or self.mutation_schedule not in ('fixed', 'linear')
                or not math.isfinite(self.mutation_sigma_end)
                or self.mutation_sigma_end <= 0
                or (self.mutation_schedule == 'linear' and
                    self.mutation_sigma_end > self.mutation_sigma)):
            raise ValueError('invalid evolution configuration')


def shapes(board, architecture='legacy'):
    """Parameter layout; the default preserves historical callers' legacy shape."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f'unknown architecture: {architecture}')
    if architecture == 'cnn':
        return evolution_cnn.shapes(board)
    if architecture == 'linear':
        return {'linear_w': (math.prod(observation_shape(board)), 3),
                'linear_b': (3,)}
    inputs = (math.prod(observation_shape(board)) if architecture == 'legacy' else GLOBAL_FEATURES)
    inputs += LOCAL_FEATURES
    return {'w1': (inputs, HIDDEN), 'b1': (HIDDEN,),
            'w2': (HIDDEN, 3), 'b2': (3,)}


def parameter_count(board, architecture):
    return sum(math.prod(shape) for shape in shapes(board, architecture).values())


def initialise(board, rng, architecture='legacy'):
    if architecture == 'cnn':
        return evolution_cnn.initialise(board, rng)
    if architecture == 'linear':
        # Small initial logits; no engineered spatial features or privileged inputs.
        return {'linear_w': rng.normal(0, .002, shapes(board, 'linear')['linear_w']).astype(np.float32),
                'linear_b': np.zeros(3, np.float32)}
    w = {k: np.zeros(v, np.float32) for k, v in shapes(board, architecture).items()}
    w['w1'][:-LOCAL_FEATURES] = rng.normal(0, .025 if architecture == 'legacy' else .055,
                                           w['w1'][:-LOCAL_FEATURES].shape)
    w['w1'][-LOCAL_FEATURES:] = rng.normal(0, .12, w['w1'][-LOCAL_FEATURES:].shape)
    w['w2'][:] = rng.normal(0, .2, w['w2'].shape)
    return w


def copy_weights(w):
    return {k: v.copy() for k, v in w.items()}


def mutate(w, rng, cfg, sigma=None):
    sigma = cfg.mutation_sigma if sigma is None else float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('mutation sigma must be finite and positive')
    child = copy_weights(w)
    for v in child.values():
        mask = rng.random(v.shape) < cfg.mutation_rate
        v[mask] += rng.normal(0, sigma, int(mask.sum())).astype(np.float32)
    return child


def architecture_of(weights, board):
    for architecture in ARCHITECTURES:
        expected = shapes(board, architecture)
        if (set(weights) == set(expected) and all(
                np.shape(weights[k]) == shape and np.asarray(weights[k]).dtype.kind in 'buifc'
                and np.isfinite(weights[k]).all() for k, shape in expected.items())):
            return architecture
    raise ValueError('invalid evolution weights or incompatible board/architecture')


def _local_patch(obs):
    _, h, w = obs.shape
    y, x = divmod(int(obs[2].argmax()), w)
    patch = np.zeros((CHANNELS, PATCH, PATCH), dtype=np.uint8)
    patch[12] = 1
    y0, y1 = max(0, y - 2), min(h, y + 3)
    x0, x1 = max(0, x - 2), min(w, x + 3)
    patch[:, y0-y+2:y1-y+2, x0-x+2:x1-x+2] = obs[:, y0:y1, x0:x1]
    return patch.ravel().astype(np.float32)


def _global_pool(obs):
    """Spatially summarise *every* full-board cell into a 6 x 6 grid.

    Mean retains occupancy density; max identifies any occupied cell in a bin.
    Padding is explicitly marked out-of-bounds in channel 12.
    """
    side = obs.shape[1]
    padded_size = math.ceil(side / POOL) * POOL
    if padded_size == side:
        padded = obs
    else:
        padded = np.zeros((CHANNELS, padded_size, padded_size), dtype=np.uint8)
        padded[:, :side, :side] = obs
        padded[12, side:, :] = 1
        padded[12, :, side:] = 1
    stride = padded_size // POOL
    blocks = padded.reshape(CHANNELS, POOL, stride, POOL, stride)
    means = blocks.mean(axis=(2, 4), dtype=np.float32)
    maxima = blocks.max(axis=(2, 4)).astype(np.float32)
    return np.concatenate((means.ravel(), maxima.ravel()))


class EvolutionActor:
    def __init__(self, weights, board, seed):
        self.architecture = architecture_of(weights, board)
        self.weights = {k: np.asarray(v, dtype=np.float32).copy() for k, v in weights.items()}
        self.shape = observation_shape(board)
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        if obs.shape != self.shape:
            raise ValueError('observation shape mismatch')
        if self.architecture == 'cnn':
            logits = evolution_cnn.forward(obs, self.weights)
            return int(self.rng.choice(np.flatnonzero(logits == logits.max())))
        if self.architecture == 'linear':
            logits = obs.ravel().astype(np.float32) @ self.weights['linear_w'] + self.weights['linear_b']
            return int(self.rng.choice(np.flatnonzero(logits == logits.max())))
        inputs = (np.concatenate((_global_pool(obs), _local_patch(obs)))
                  if self.architecture == 'compact' else
                  np.concatenate((obs.ravel().astype(np.float32), _local_patch(obs))))
        p = self.weights
        hidden = np.maximum(0, inputs @ p['w1'] + p['b1'])
        logits = hidden @ p['w2'] + p['b2']
        return int(self.rng.choice(np.flatnonzero(logits == logits.max())))


def factory(weights, board, seed):
    """Load both compact v2 and full-board legacy v1 policy weights."""
    return EvolutionActor(weights, board, seed)


def balanced_counts(weights: dict[str, float], total: int, generation: int = 0):
    """Largest-remainder allocation with deterministic rotating tie breaks."""
    if total < 1 or not weights or any(w < 0 or not math.isfinite(w) for w in weights.values()):
        raise ValueError('invalid opponent quota')
    labels = list(weights)
    mass = sum(weights.values())
    if mass <= 0:
        raise ValueError('opponent quota must be positive')
    ideal = [total * weights[name] / mass for name in labels]
    counts = [int(math.floor(n)) for n in ideal]
    remaining = total - sum(counts)
    ranked = sorted(range(len(labels)),
                    key=lambda i: (-(ideal[i]-counts[i]), (i-generation) % len(labels)))
    for i in ranked[:remaining]:
        counts[i] += 1
    return dict(zip(labels, counts))


def make_pair_schedule(board, pool, open_rng, fitness_pairs, generation,
                       survival_safe_fraction=.5, fitness_sampling='balanced'):
    # Generate opening seeds independently of the opponent schedule. This
    # keeps the same opening RNG sequence in random-versus-balanced ablations.
    openings = []
    for _ in range(fitness_pairs):
        board_seed = open_rng.getrandbits(63)
        actor_seed = open_rng.getrandbits(63)
        opponent_seed = open_rng.getrandbits(63)
        axis = open_rng.choice(('horizontal', 'vertical'))
        openings.append(tuple(Opening(board_seed, axis, False, side,
                                      actor_seed, opponent_seed) for side in (0, 1)))
    # The *same explicit matchups* and paired seeds are reused for all
    # candidates. Fresh openings vary between generations, opponent mix not.
    if fitness_sampling == 'balanced':
        if pool.stage == 'selfplay' and pool.history:
            weights = {k: .6*v for k, v in STAGES['selfplay'].items()}
            weights['history'] = .4
        elif pool.stage == 'survival':
            weights = {'random': 1. - survival_safe_fraction,
                       'safe': survival_safe_fraction}
        else:
            weights = STAGES[pool.stage]
        counts = balanced_counts(weights, fitness_pairs, generation)
        labels = [name for name, n in counts.items() for _ in range(n)]
        pool.rng.shuffle(labels)
    else:
        labels = [pool.sample(board, opening[0].opponent_seed)[0]
                  for opening in openings]
    pairs = []
    for pair_index, (name, pair) in enumerate(zip(labels, openings)):
        historical = next((f for f in pool.history if f.name == name), None)
        if name == 'history':
            # Choose an actual frozen checkpoint once for this pair.
            mode = pool.rng.choice(('best', 'latest', 'uniform'))
            historical = (max(pool.history, key=lambda f: f.score) if mode == 'best' else
                          pool.history[-1] if mode == 'latest' else pool.rng.choice(pool.history))
            name = historical.name
        if historical is None and name not in STAGES['selfplay']:
            raise ValueError(f'unknown opponent: {name}')
        pairs.append((pair_index, name, historical, pair))
    return pairs


class EvolutionLearner:
    algorithm = 'evolution'

    def __init__(self, board, seed=2026, cfg=None, initial_weights=None):
        self.cfg = cfg or EvolutionConfig()
        self.rng = np.random.default_rng(seed)
        if initial_weights is not None:
            actor = factory(initial_weights, board, seed)
            if actor.architecture != self.cfg.architecture:
                raise ValueError(f'warm-start architecture is {actor.architecture}; '
                                 f'use --architecture {actor.architecture}')
            first = actor.weights
        else:
            first = initialise(board, self.rng, self.cfg.architecture)
        self.population = [copy_weights(first)]
        self.population.extend(initialise(board, self.rng, self.cfg.architecture)
                               if initial_weights is None else mutate(first, self.rng, self.cfg)
                               for _ in range(self.cfg.population-1))
        self.champion = copy_weights(first)
        self.generation = 0
        self.fitness = None
        self.last_generation_record = None
        self.parameter_count = parameter_count(board, self.cfg.architecture)
        self.last_mutation_sigma = None

    def mutation_sigma_at(self, progress: float) -> float:
        """Sigma applied when creating the *next* generation's offspring."""
        if not math.isfinite(progress):
            raise ValueError('training progress must be finite')
        fraction = min(1.0, max(0.0, float(progress)))
        if self.cfg.mutation_schedule == 'fixed':
            return self.cfg.mutation_sigma
        return (self.cfg.mutation_sigma + fraction *
                (self.cfg.mutation_sigma_end - self.cfg.mutation_sigma))

    @property
    def batch_episodes(self):
        return self.cfg.population * self.cfg.fitness_pairs * 2

    def snapshot(self):
        return copy_weights(self.champion)

    def diagnostics(self):
        return dict(
            generation=self.generation, population=self.cfg.population,
            architecture=self.cfg.architecture, parameters=self.parameter_count,
            fitness_sampling=self.cfg.fitness_sampling,
            survival_safe_fraction=self.cfg.survival_safe_fraction,
            mutation_schedule=self.cfg.mutation_schedule,
            last_mutation_sigma=self.last_mutation_sigma,
            best_fitness=None if self.fitness is None else float(self.fitness.max()),
            mean_fitness=None if self.fitness is None else float(self.fitness.mean()),
            worst_fitness=None if self.fitness is None else float(self.fitness.min()),
            fitness_std=None if self.fitness is None else float(self.fitness.std()),
            opponent_pairs={} if self.last_generation_record is None
            else self.last_generation_record['opponent_pairs'])

    def _pair_schedule(self, board, pool, open_rng):
        return make_pair_schedule(board, pool, open_rng, self.cfg.fitness_pairs, self.generation,
                                  self.cfg.survival_safe_fraction, self.cfg.fitness_sampling)

    def train_batch(self, board, pool, open_rng, *, progress=0.):
        pairs = self._pair_schedule(board, pool, open_rng)
        rows, fitness = [], []
        opponent_rewards = {name: [] for _, name, _, _ in pairs}
        candidate_opponent_rewards = []
        for i, weights in enumerate(self.population):
            per_opponent = {name: [] for name in opponent_rewards}
            rewards = []
            for pair_index, name, historical, openings in pairs:
                for opening in openings:
                    if historical is None:
                        opponent = ScriptedPolicy(name, seed=opening.opponent_seed)
                    else:
                        opponent = ObservationPolicy(historical.factory(
                            copy_weights(historical.weights), board, opening.opponent_seed))
                    result = play_episode(board, opening,
                                          factory(weights, board, opening.actor_seed), opponent)
                    rewards.append(result.reward)
                    opponent_rewards[name].append(result.reward)
                    per_opponent[name].append(result.reward)
                    rows.append(dict(generation=self.generation+1, candidate=i,
                                     pair_index=pair_index, opponent=name,
                                     opening=asdict(opening), **asdict(result)))
            fitness.append(float(np.mean(rewards)))
            candidate_opponent_rewards.append(per_opponent)
        self.fitness = np.array(fitness)
        self.generation += 1
        order = np.lexsort((self.rng.random(len(fitness)), -self.fitness))
        elite = [copy_weights(self.population[i]) for i in order[:self.cfg.elites]]
        self.champion = copy_weights(elite[0])
        sigma = self.mutation_sigma_at(progress)
        self.last_mutation_sigma = sigma
        self.population = elite + [mutate(elite[int(self.rng.integers(len(elite)))],
                                          self.rng, self.cfg, sigma=sigma)
                                   for _ in range(self.cfg.population-len(elite))]
        pair_counts = dict(Counter(name for _, name, _, _ in pairs))
        champion_outcomes = candidate_opponent_rewards[int(order[0])]
        champion_breakdown = {name: dict(
            wins=sum(r > 0 for r in rewards), draws=sum(r == 0 for r in rewards),
            losses=sum(r < 0 for r in rewards), games=len(rewards),
            mean_reward=float(np.mean(rewards)))
            for name, rewards in champion_outcomes.items() if rewards}
        per_opponent_fitness = {name: dict(
            mean=float(np.mean([np.mean(rewards[name]) for rewards in candidate_opponent_rewards])),
            std=float(np.std([np.mean(rewards[name]) for rewards in candidate_opponent_rewards])))
            for name in opponent_rewards}
        self.last_generation_record = dict(
            generation=self.generation, stage=pool.stage, fitness_sampling=self.cfg.fitness_sampling,
            survival_safe_fraction=self.cfg.survival_safe_fraction,
            mutation_schedule=self.cfg.mutation_schedule,
            training_progress_at_generation_start=float(progress),
            next_offspring_mutation_sigma=float(sigma),
            opponent_pairs=pair_counts, population=self.cfg.population,
            fitness_pairs=self.cfg.fitness_pairs, total_games=len(rows),
            best_fitness=float(self.fitness.max()), mean_fitness=float(self.fitness.mean()),
            worst_fitness=float(self.fitness.min()), fitness_std=float(self.fitness.std()),
            best_candidate=int(order[0]), elite_indices=[int(x) for x in order[:self.cfg.elites]],
            opponent_mean_reward={name: float(np.mean(rewards)) for name, rewards in opponent_rewards.items()},
            opponent_fitness_distribution=per_opponent_fitness,
            best_candidate_by_opponent=champion_breakdown)
        return rows
