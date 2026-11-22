"""Phase 5.5: teacher-assisted, CPU-friendly enhanced DQN and PPO campaigns.

Commands: collect, pretrain, train, campaign. See README_PHASE55.md.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from dataclasses import asdict
from tron_ai.engine import TronConfig
from tron_ai.training.checkpoints import load
from tron_ai.training.runner import RunConfig,run
from tron_ai.training.teacher import collect,pretrain


def _weights(path,board,algorithm):
    if path is None:return None
    return load(Path(path),config=board,algorithm=algorithm)[0]


def _train(algorithm,board,seconds,interval,pairs,seed,source,out,threads=1,resume=None,max_episodes=None):
    import torch
    torch.set_num_threads(threads)
    if algorithm=='dqn':
        from tron_ai.training.advanced_dqn import EnhancedDQNLearner,EnhancedDQNConfig,factory,ARCHITECTURE
        cfg=EnhancedDQNConfig();learner=EnhancedDQNLearner(board,seed,cfg,_weights(source,board,algorithm))
    else:
        from tron_ai.training.advanced_ppo import EnhancedPPOLearner,EnhancedPPOConfig
        from tron_ai.training.ppo import factory,ARCHITECTURE
        cfg=EnhancedPPOConfig();learner=EnhancedPPOLearner(board,seed,cfg,_weights(source,board,algorithm))
    spec=dict(**asdict(cfg),architecture=ARCHITECTURE,training_variant='teacher-assisted-league-v1',
              pretrained_from=str(source) if source else None,torch=torch.__version__,threads=threads,
              initialization='weights-only warm start' if source else 'random weights')
    return run(learner,factory,board,RunConfig(seconds,interval,pairs,seed,'league',max_episodes),Path(out),
               algorithm_config=spec,resume_state_path=resume,persist_training_state=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect',help='collect same-observation teacher actions')
    c.add_argument('--examples',type=int,default=8000);c.add_argument('--seed',type=int,default=2026)
    c.add_argument('--width',type=int,default=30);c.add_argument('--height',type=int,default=24)
    c.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('pretrain',help='supervised imitation of scripted teacher actions')
    c.add_argument('--dataset',type=Path,required=True)
    c.add_argument('--algorithm',choices=['dqn','ppo','both'],default='both')
    c.add_argument('--steps',type=int,default=800);c.add_argument('--seed',type=int,default=2026)
    c.add_argument('--threads',type=int,default=1)
    c.add_argument('--dqn-init',type=Path);c.add_argument('--ppo-init',type=Path)
    c.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('train',help='enhanced RL with full-state checkpointing')
    c.add_argument('algorithm',choices=['dqn','ppo'])
    c.add_argument('--seconds',type=float,default=3600)
    c.add_argument('--eval-interval',type=float,default=600)
    c.add_argument('--eval-pairs',type=int,default=8)
    c.add_argument('--seed',type=int,default=2026);c.add_argument('--threads',type=int,default=1)
    c.add_argument('--width',type=int,default=30);c.add_argument('--height',type=int,default=24)
    c.add_argument('--init-from',type=Path);c.add_argument('--resume-from',type=Path)
    c.add_argument('--max-episodes',type=int);c.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('campaign',help='collect, pretrain and train both algorithms automatically')
    c.add_argument('--seconds',type=float,default=3600,help='active RL seconds PER algorithm')
    c.add_argument('--examples',type=int,default=8000,help='number of teacher action labels')
    c.add_argument('--imitation-steps',type=int,default=800,help='supervised gradient steps PER algorithm')
    c.add_argument('--eval-interval',type=float,default=600)
    c.add_argument('--eval-pairs',type=int,default=8)
    c.add_argument('--seed',type=int,default=2026);c.add_argument('--threads',type=int,default=1)
    c.add_argument('--width',type=int,default=30);c.add_argument('--height',type=int,default=24)
    c.add_argument('--dqn-init',type=Path);c.add_argument('--ppo-init',type=Path)
    c.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='collect':
        print(json.dumps(collect(TronConfig(a.width,a.height),a.examples,a.seed,a.out),indent=2))
    elif a.command=='pretrain':
        from tron_ai.training.teacher import load_dataset
        _,_,_,meta=load_dataset(a.dataset)
        board=TronConfig(**meta['board'])
        result={}
        for alg in ('dqn','ppo') if a.algorithm=='both' else (a.algorithm,):
            result[alg]=pretrain(a.dataset,alg,a.steps,a.seed,a.out/alg,threads=a.threads,
                                 initial_weights=_weights(getattr(a,alg+'_init'),board,alg))
        print(json.dumps(result,indent=2))
    elif a.command=='train':
        if a.init_from is not None and a.resume_from is not None:
            p.error('use --init-from OR --resume-from, not both')
        board=TronConfig(a.width,a.height)
        print(json.dumps(_train(a.algorithm,board,a.seconds,a.eval_interval,a.eval_pairs,a.seed,
             a.init_from,a.out,a.threads,a.resume_from,a.max_episodes),indent=2))
    elif a.command=='campaign':
        board=TronConfig(a.width,a.height)
        root=a.out
        if root.exists():raise FileExistsError(f'campaign output already exists: {root}')
        root.mkdir(parents=True)
        teacher_file=root/'teacher_data.npz'
        print('Phase 5.5: collecting teacher examples',flush=True)
        teacher=collect(board,a.examples,a.seed,teacher_file)
        summary=dict(teacher=teacher,models={})
        for alg in ('dqn','ppo'):
            print(f'Phase 5.5: {alg.upper()} teacher pretraining',flush=True)
            source=_weights(getattr(a,alg+'_init'),board,alg)
            warm=pretrain(teacher_file,alg,a.imitation_steps,a.seed,root/'imitation'/alg,
                          threads=a.threads,initial_weights=source)
            chosen=root/'imitation'/alg/'best.npz'
            safeguard=None
            if source is not None:
                # Development-set win-rate safeguard protects a strong existing
                # checkpoint from an overaggressive supervised warm start.
                from tron_ai.training.benchmark import schedule,evaluate
                if alg=='dqn':
                    from tron_ai.training.advanced_dqn import upgrade_legacy_weights,factory
                    original=upgrade_legacy_weights(source)
                else:
                    from tron_ai.training.ppo import factory
                    original=source
                dev_cases=schedule(8,seed=2026055) # never validation or holdout seed
                score_old=evaluate(factory,original,board,dev_cases)[0]['score']
                score_teacher=evaluate(factory,_weights(chosen,board,alg),board,dev_cases)[0]['score']
                if score_teacher+1/64 < score_old:
                    chosen=getattr(a,alg+'_init')
                safeguard=dict(development_seed=2026055,initial_score=score_old,
                               imitation_score=score_teacher,
                               selected='imitation' if chosen==root/'imitation'/alg/'best.npz' else 'previous',
                               note='development-selection benchmark; not independent holdout')
                print(f'{alg.upper()} selection: previous={score_old:.1%}, imitation={score_teacher:.1%}; '
                      f'using {safeguard["selected"]}',flush=True)
            print(f'Phase 5.5: {alg.upper()} reinforcement learning',flush=True)
            rl=_train(alg,board,a.seconds,a.eval_interval,a.eval_pairs,a.seed,
                      chosen,root/(alg+'_rl'),a.threads)
            summary['models'][alg]=dict(imitation=warm,safeguard=safeguard,reinforcement=rl)
            (root/'campaign_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
