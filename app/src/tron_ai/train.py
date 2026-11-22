"""Train and evaluate DQN, PPO or evolutionary policies (GA, ES, ARS)."""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
from tron_ai.engine import TronConfig
from tron_ai.training import reference
from tron_ai.training.runner import RunConfig, run, write_json
from tron_ai.training.benchmark import evaluate, schedule, HOLDOUT_SEED, VALIDATION_SEED
from tron_ai.training.checkpoints import load
from tron_ai.training.pool import STAGES


def policy_factory(algorithm,threads=1):
    if algorithm=='reference': return reference.factory
    if algorithm=='evolution':
        from tron_ai.training import evolution
        return evolution.factory
    if algorithm in ('dqn','ppo'):
        import torch
        from tron_ai.training import dqn, ppo
        if threads<1: raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        return dqn.factory if algorithm=='dqn' else ppo.factory
    raise ValueError(f'unsupported algorithm: {algorithm}')


def parse_curriculum(value):
    try:
        stages=tuple((name,float(fraction)) for name,fraction in (part.split(':') for part in value.split(',')))
        RunConfig(stage=stages[0][0],curriculum=stages)
        return stages
    except (ValueError,IndexError) as exc:
        raise argparse.ArgumentTypeError('expected stage:fraction,... with fractions summing to 1') from exc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('reference','dqn','ppo','evolution'):
        p=sub.add_parser(name,help=f'train {name} policy')
        p.add_argument('--seconds',type=float,default=30)
        p.add_argument('--eval-interval',type=float,default=10)
        p.add_argument('--eval-pairs',type=int,default=2)
        p.add_argument('--seed',type=int,default=2026)
        p.add_argument('--stage',choices=STAGES,default='survival')
        p.add_argument('--curriculum',type=parse_curriculum,help='continuous schedule, e.g. survival:0.5,space:0.5; overrides --stage')
        p.add_argument('--width',type=int,default=30); p.add_argument('--height',type=int,default=24)
        p.add_argument('--max-episodes',type=int)
        p.add_argument('--out',type=Path,required=True)
        if name=='dqn':
            p.add_argument('--threads',type=int,default=1)
            for k,v in [('capacity',16384),('batch-size',64),('warmup',512),('update-every',4),('target-every',250),('epsilon-decay',30000)]:
                p.add_argument('--'+k,type=int,default=v)
            for k,v in [('lr',.0005),('gamma',.99),('epsilon-start',1.),('epsilon-end',.05)]:
                p.add_argument('--'+k,type=float,default=v)
            p.add_argument('--init-from',type=Path,help='weights-only warm start; resets optimizer, replay and epsilon')
        if name=='ppo':
            p.add_argument('--threads',type=int,default=1)
            for k,v in [('rollout-steps',512),('batch-size',64),('epochs',4)]:
                p.add_argument('--'+k,type=int,default=v)
            for k,v in [('lr',.0003),('gamma',.99),('gae-lambda',.95),('clip',.2),('entropy-coef',.02),('value-coef',.5),('max-grad-norm',.5)]:
                p.add_argument('--'+k,type=float,default=v)
            p.add_argument('--init-from',type=Path,help='weights-only warm start; resets optimizer and rollout')
        if name=='evolution':
            for k,v in [('population',16),('elites',4),('fitness-pairs',4)]:
                p.add_argument('--'+k,type=int,default=v)
            for k,v in [('mutation-sigma',.08),('mutation-rate',.1)]:
                p.add_argument('--'+k,type=float,default=v)
            p.add_argument('--architecture', choices=('compact','legacy','cnn','linear'), default='compact',
                           help='compact, legacy dense, CNN or full-board linear')
            p.add_argument('--fitness-sampling',choices=('balanced','random'),default='balanced',
                           help='balanced fixed opponent quotas per generation or legacy random draw')
            p.add_argument('--survival-safe-fraction',type=float,default=.5,
                           help='proportion of Random Safe in balanced survival fitness (0 to 1)')
            p.add_argument('--mutation-schedule',choices=('fixed','linear'),default='fixed',
                           help='fixed sigma or linearly decrease sigma over training budget')
            p.add_argument('--mutation-sigma-end',type=float,default=.005,
                           help='final sigma for linear schedule (must not exceed start)')
            p.add_argument('--optimizer',choices=('genetic','es','ars'),default='genetic',
                           help='elitist GA (default), antithetic ES, or augmented random search')
            p.add_argument('--directions',type=int,default=12,
                           help='number of mirrored perturbation pairs per ES/ARS iteration')
            p.add_argument('--top-directions',type=int,default=6,
                           help='best ARS directions retained for each update')
            p.add_argument('--noise-std',type=float,default=.025,
                           help='standard deviation of ES/ARS parameter perturbations')
            p.add_argument('--step-size',type=float,default=.01,
                           help='ES/ARS parameter update coefficient')
            p.add_argument('--max-update-l2',type=float,default=1.,
                           help='stability cap on L2 norm of ES/ARS parameter update')
            p.add_argument('--init-from',type=Path,help='weights-only warm start; resets optimiser/population')
    p=sub.add_parser('evaluate',help='evaluate a saved policy without training')
    p.add_argument('checkpoint',type=Path); p.add_argument('--pairs',type=int,default=10)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--split',choices=['validation','holdout'],default='holdout')
    p.add_argument('--holdout-seed',type=int,default=HOLDOUT_SEED,help='independent seed; ignored for validation')
    p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('plot',help='plot training history')
    p.add_argument('run',type=Path)
    args=parser.parse_args(argv)
    if args.command in ('reference','dqn','ppo','evolution'):
        board=TronConfig(args.width,args.height)
        factory=policy_factory(args.command,getattr(args,'threads',1))
        if args.command=='reference':
            learner=reference.ReferenceLearner(args.seed); spec={'learning_rate':.05}
        elif args.command=='dqn':
            import torch
            from tron_ai.training import dqn
            cfg=dqn.DQNConfig(**{k:getattr(args,k) for k in asdict(dqn.DQNConfig())})
            weights=load(args.init_from,config=board,algorithm='dqn')[0] if args.init_from else None
            learner=dqn.DQNLearner(board,args.seed,cfg,weights)
            spec={**asdict(cfg),'architecture':dqn.ARCHITECTURE,'torch':str(torch.__version__),'device':'cpu','threads':args.threads,'init_from':str(args.init_from) if args.init_from else None}
        elif args.command=='ppo':
            import torch
            from tron_ai.training import ppo
            cfg=ppo.PPOConfig(**{k:getattr(args,k) for k in asdict(ppo.PPOConfig())})
            weights=load(args.init_from,config=board,algorithm='ppo')[0] if args.init_from else None
            learner=ppo.PPOLearner(board,args.seed,cfg,weights)
            spec={**asdict(cfg),'architecture':ppo.ARCHITECTURE,'torch':str(torch.__version__),'device':'cpu','threads':args.threads,'init_from':str(args.init_from) if args.init_from else None,'evaluation_policy':'seeded categorical sampling'}
        else:
            from tron_ai.training import evolution
            weights=load(args.init_from,config=board,algorithm='evolution')[0] if args.init_from else None
            if args.optimizer=='genetic':
                cfg=evolution.EvolutionConfig(**{k:getattr(args,k) for k in asdict(evolution.EvolutionConfig())})
                learner=evolution.EvolutionLearner(board,args.seed,cfg,weights)
            else:
                from tron_ai.training import search
                cfg=search.SearchConfig(**{k:getattr(args,k) for k in asdict(search.SearchConfig())})
                learner=search.SearchLearner(board,args.seed,cfg,weights)
            spec={**asdict(cfg),'optimizer':args.optimizer,
                  'architecture':evolution.ARCHITECTURES[cfg.architecture],
                  'parameter_count':evolution.parameter_count(board,cfg.architecture),
                  'device':'cpu','init_from':str(args.init_from) if args.init_from else None,
                  'fitness':'mean terminal reward on common paired games; mirrored matched conditions for ES/ARS' }
        result=run(learner,factory,board,RunConfig(args.seconds,args.eval_interval,args.eval_pairs,args.seed,args.curriculum[0][0] if args.curriculum else args.stage,args.max_episodes,curriculum=args.curriculum or ()),args.out,algorithm_config=spec)
        print(json.dumps(result,indent=2))
    elif args.command=='evaluate':
        weights,meta=load(args.checkpoint); board=TronConfig(**meta['board'])
        load(args.checkpoint,config=board,algorithm=meta['algorithm'])
        factory=policy_factory(meta['algorithm'],args.threads)
        seed=meta['run']['validation_seed'] if args.split=='validation' else args.holdout_seed
        metrics,rows=evaluate(factory,weights,board,schedule(args.pairs,seed))
        args.out.mkdir(parents=True,exist_ok=True)
        if any(args.out.iterdir()): raise FileExistsError(f'output directory not empty: {args.out}')
        write_json(args.out/'evaluation.json',dict(checkpoint=str(args.checkpoint),split=args.split,seed=seed,pairs=args.pairs,board=meta['board'],metrics=metrics,rows=rows))
        print(json.dumps(metrics,indent=2))
    else:
        from tron_ai.training.plots import plot
        print(plot(args.run))

if __name__=='__main__': main()
