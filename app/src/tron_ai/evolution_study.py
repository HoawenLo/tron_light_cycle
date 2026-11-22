"""Serial, reproducible neuroevolution ablation experiments; never uses holdout seeds.

Run `python -m tron_ai.evolution_study --help` for variants and usage.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from .training.runner import write_json

# Each successive row isolates one changed factor from the preceding row,
# except 'tuned_legacy' (both mutation rate and sigma change together).
VARIANTS = {
    'legacy_random': dict(architecture='legacy', fitness_sampling='random',
                          fitness_pairs=4, mutation_rate=.10, mutation_sigma=.08),
    'legacy_balanced': dict(architecture='legacy', fitness_sampling='balanced',
                            fitness_pairs=4, mutation_rate=.10, mutation_sigma=.08),
    'legacy_more_pairs': dict(architecture='legacy', fitness_sampling='balanced',
                              fitness_pairs=16, mutation_rate=.10, mutation_sigma=.08),
    'tuned_legacy': dict(architecture='legacy', fitness_sampling='balanced',
                         fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02),
    'tuned_compact': dict(architecture='compact', fitness_sampling='balanced',
                          fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02),
    # Phase 5.3: isolate the representation; use identical evolution settings.
    'tuned_cnn': dict(architecture='cnn', fitness_sampling='balanced',
                      fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02),
    # Phase 5.4: optimiser-by-representation comparison on the same game,
    # timing budget, evaluation and terminal reward. No holdout used for selection.
    'cpu_ga_legacy': dict(optimizer='genetic', architecture='legacy',
                           fitness_sampling='balanced', fitness_pairs=8,
                           mutation_rate=.01, mutation_sigma=.02),
    'cpu_es_legacy': dict(optimizer='es', architecture='legacy',
                           fitness_pairs=8, directions=12, top_directions=6,
                           noise_std=.025, step_size=.01),
    'cpu_ars_linear': dict(optimizer='ars', architecture='linear',
                            fitness_pairs=8, directions=12, top_directions=6,
                            noise_std=.025, step_size=.01),
    'cpu_es_linear': dict(optimizer='es', architecture='linear',
                           fitness_pairs=8, directions=12, top_directions=6,
                           noise_std=.025, step_size=.01),
    # Phase 5.2 factorial ablation: keep legacy architecture and fitness-pairs
    # constant so opponent composition and sigma schedule are isolated.
    'control_50_fixed': dict(architecture='legacy', fitness_sampling='balanced',
                             fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02,
                             survival_safe_fraction=.5, mutation_schedule='fixed'),
    'safe80_fixed': dict(architecture='legacy', fitness_sampling='balanced',
                         fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02,
                         survival_safe_fraction=.8, mutation_schedule='fixed'),
    'control_50_adaptive': dict(architecture='legacy', fitness_sampling='balanced',
                                fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02,
                                survival_safe_fraction=.5, mutation_schedule='linear',
                                mutation_sigma_end=.005),
    'safe80_adaptive': dict(architecture='legacy', fitness_sampling='balanced',
                            fitness_pairs=16, mutation_rate=.01, mutation_sigma=.02,
                            survival_safe_fraction=.8, mutation_schedule='linear',
                            mutation_sigma_end=.005),
}


def report(out: Path):
    manifest = json.loads((out/'manifest.json').read_text())
    rows = []
    for entry in manifest['runs']:
        path = out/entry['directory']
        summary = json.loads((path/'summary.json').read_text())
        evaluations = [json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()]
        best = next(r for r in evaluations if r['checkpoint'] == summary['best_checkpoint'])
        row = dict(variant=entry['variant'], seed=entry['seed'],
                   train_seconds=summary['training_seconds'],
                   evaluation_seconds=summary['evaluation_seconds'],
                   steps=summary['environment_steps'], episodes=summary['episodes'],
                   generations=evaluations[-1].get('diagnostics', {}).get('generation'),
                   best_generation=best.get('diagnostics', {}).get('generation'),
                   initial_score=evaluations[0]['metrics']['score'],
                   final_score=evaluations[-1]['metrics']['score'],
                   best_score=best['metrics']['score'],
                   best_checkpoint=summary['best_checkpoint'])
        row.update({f'{name}_win_rate': m['win_rate'] for name, m in
                    best['metrics']['by_opponent'].items()})
        rows.append(row)
    if not rows:
        raise ValueError('no completed experiments')
    with (out/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    # Cross-seed means are descriptive; one seed should never be interpreted as
    # an estimate of training reliability or algorithmic superiority.
    summary = []
    import statistics
    for variant in dict.fromkeys(r['variant'] for r in rows):
        subset = [r for r in rows if r['variant'] == variant]
        scores = [r['best_score'] for r in subset]
        safe = [r.get('safe_win_rate', 0.) for r in subset]
        summary.append(dict(variant=variant, n_seeds=len(subset),
                            mean_best_score=statistics.mean(scores),
                            sd_best_score=(statistics.stdev(scores) if len(scores)>1 else None),
                            mean_safe_win_rate=statistics.mean(safe),
                            mean_actual_training_seconds=statistics.mean(r['train_seconds'] for r in subset)))
    with (out/'aggregate_by_variant.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return rows  # CSV is always generated; plotting is optional.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    names = list(dict.fromkeys(r['variant'] for r in rows))
    for index, variant in enumerate(names):
        subset = [r for r in rows if r['variant'] == variant]
        for r in subset:
            axes[0].scatter(index, r['best_score'], label=(str(r['seed']) if index == 0 else None))
            axes[1].scatter(index, r['steps'])
    for ax, label in zip(axes, ('Best fixed-validation score', 'Environment steps')):
        ax.set_xticks(range(len(names)), names, rotation=25, ha='right')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_ylim(0, 1)
    fig.suptitle('Neuroevolution ablation | equal requested training time per run')
    fig.savefig(out/'comparison.png', dpi=150)
    plt.close(fig)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--eval-interval', type=float, default=60)
    p.add_argument('--eval-pairs', type=int, default=8)
    p.add_argument('--population', type=int, default=16)
    p.add_argument('--elites', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[2026])
    p.add_argument('--stage', choices=('survival', 'space', 'competitive', 'selfplay'),
                   default='survival')
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS),
                   default=['control_50_fixed', 'safe80_fixed',
                            'control_50_adaptive', 'safe80_adaptive'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args(argv)
    out = args.out.resolve()
    if args.report_only:
        print(json.dumps(report(out), indent=2))
        return
    if (args.seconds <= 0 or args.eval_interval <= 0 or args.eval_pairs < 1 or
            args.population < 2 or not 1 <= args.elites < args.population or
            any(seed < 0 for seed in args.seeds) or
            len(set(args.seeds)) != len(args.seeds) or
            len(set(args.variants)) != len(args.variants)):
        p.error('invalid settings, duplicates, or training budget')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'output directory not empty: {out}')
    env = os.environ.copy()
    env.update({name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
                                      'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    runs = []
    for seed in args.seeds:
        for variant in args.variants:
            path = out/f'{variant}_seed{seed}'
            config = VARIANTS[variant]
            command = [sys.executable, '-m', 'tron_ai.train', 'evolution',
                       '--seconds', str(args.seconds), '--eval-interval', str(args.eval_interval),
                       '--eval-pairs', str(args.eval_pairs), '--population', str(args.population),
                       '--elites', str(args.elites), '--stage', args.stage,
                       '--seed', str(seed), '--out', str(path)]
            for key, value in config.items():
                command += ['--'+key.replace('_','-'), str(value)]
            runs.append(dict(seed=seed, variant=variant, directory=path.name,
                             config=config, command=command))
    write_json(out/'manifest.json', dict(runs=runs, stage=args.stage,
                                        seconds=args.seconds, eval_interval=args.eval_interval,
                                        eval_pairs=args.eval_pairs,
                                        selection='best fixed validation; no holdout used'))
    for entry in runs:
        print(f"Running {entry['variant']} seed={entry['seed']}", flush=True)
        with (out/(entry['directory']+'.log')).open('w') as handle:
            result = subprocess.run(entry['command'], env=env, text=True,
                                    stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{entry['variant']} failed: see {entry['directory']}.log")
        print(f"  finished: {entry['directory']}", flush=True)
    print(json.dumps(report(out), indent=2))


if __name__ == '__main__':
    main()
