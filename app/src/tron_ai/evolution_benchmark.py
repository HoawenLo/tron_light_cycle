"""Compare raw policy inference latency on identical cached Tron observations.

Run `python -m tron_ai.evolution_benchmark --samples 300 --repeats 3`.
Does not measure game time, learning speed, or final playing strength.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from tron_ai.engine import TronConfig, TronEnv
from tron_ai.training.evolution import ARCHITECTURES, factory, initialise, parameter_count


def observations(board, n, seed):
    """Collect a repeatable sequence of valid game states outside timed loops."""
    rng = np.random.default_rng(seed)
    env = TronEnv(board, seed=seed)
    items = []
    for _ in range(n):
        items.append(env.observe(0))
        if env.state.done:
            env.reset(seed=int(rng.integers(0, 2**31)), mode='random_mirrored')
        step = env.step((int(rng.integers(3)), int(rng.integers(3))))
        if step.done:
            env.reset(seed=int(rng.integers(0, 2**31)), mode='random_mirrored')
    return items


def benchmark(board, architectures=('legacy', 'cnn'), samples=300, repeats=3, seed=2026):
    if samples < 1 or repeats < 1 or not architectures:
        raise ValueError('samples, repeats, and architectures must be nonempty')
    if any(name not in ARCHITECTURES for name in architectures):
        raise ValueError('unknown architecture')
    obs = observations(board, samples, seed)
    rows = []
    for name in architectures:
        actor = factory(initialise(board, np.random.default_rng(seed), name), board, seed)
        for item in obs[:min(20, len(obs))]:
            actor.act(item)  # warm up BLAS/NumPy before timing
        elapsed = []
        for _ in range(repeats):
            tic = time.perf_counter()
            for item in obs:
                actor.act(item)
            elapsed.append(time.perf_counter() - tic)
        median = statistics.median(elapsed)
        rows.append(dict(architecture=name, parameters=parameter_count(board, name),
                         observations=samples, repeats=repeats,
                         median_us_per_action=1e6*median/samples,
                         actions_per_second=samples/median,
                         runs_seconds=elapsed))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--width', type=int, default=30)
    parser.add_argument('--height', type=int, default=24)
    parser.add_argument('--architectures', nargs='+', choices=list(ARCHITECTURES), default=['legacy','cnn'])
    parser.add_argument('--out', type=Path, help='optional JSON results file')
    args = parser.parse_args(argv)
    results = benchmark(TronConfig(args.width,args.height), args.architectures,
                        args.samples,args.repeats,args.seed)
    print(json.dumps(results, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise FileExistsError(f'output file exists: {args.out}')
        args.out.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
