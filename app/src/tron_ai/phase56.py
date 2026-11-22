"""Phase 5.6: straightforward scripted-opponent RL with stable DQN/PPO fine-tuning.

Usage: python -m tron_ai.phase56 train {dqn,ppo} --init-from OLD.npz --out NEW
No teacher collection, supervised learning, or policy cloning of scripted bots.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import torch
from tron_ai.engine import TronConfig
from tron_ai.opponents import ScriptedPolicy
from tron_ai.training.checkpoints import load
from tron_ai.training.core import play_episode, training_opening
from tron_ai.training.runner import RunConfig, run
from tron_ai.training.stable_dqn import StableDQNConfig, StableDQNLearner
from tron_ai.training.stable_ppo import StablePPOConfig, StablePPOLearner
from tron_ai.training.advanced_dqn import factory as dqn_factory, ARCHITECTURE as DQN_ARCH
from tron_ai.training.dqn import factory as compatible_dqn_factory
from tron_ai.training.ppo import factory as ppo_factory, ARCHITECTURE as PPO_ARCH


def preload_original_dqn_games(learner, board, source, *, games=160, seed=2026):
    """Fill a permanent replay archive with the starting agent's actual game outcomes.

    The actor chooses its own actions; opponents use their normal scripted policies.
    This does not update weights, does not ask for teacher actions, and does not
    charge these preliminary games to the requested RL time budget.
    """
    if games < 0:
        raise ValueError('games must not be negative')
    rng=random.Random(seed+560056)
    actor=compatible_dqn_factory(source,board,seed+1)
    distribution={'safe':.50,'space':.40,'territory':.10}
    results={name:dict(episodes=0,wins=0,draws=0,losses=0,steps=0) for name in distribution}
    for i in range(games):
        opening=training_opening(rng,i)
        name=rng.choices(list(distribution),list(distribution.values()))[0]
        outcome=play_episode(board,opening,actor,ScriptedPolicy(name,seed=opening.opponent_seed),
                             update=learner.record_protected)
        row=results[name]
        row['episodes']+=1;row['steps']+=outcome.steps
        row['wins' if outcome.reward==1 else 'draws' if outcome.reward==0 else 'losses']+=1
    return dict(games=games,distribution=distribution,protected_replay_size=learner.protected.size,
                scripted_opponents=results)


def _resume_weights(path, alg):
    # Only for constructing networks before importing the complete saved state.
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('format_version')!=1 or data.get('algorithm')!=alg:
        raise ValueError('incompatible full-state checkpoint')
    field='online' if alg=='dqn' else 'network'
    return {k:v.detach().numpy().copy() for k,v in data['learner'][field].items()}


def train(args):
    if args.init_from is not None and args.resume_from is not None:
        raise ValueError('use --init-from OR --resume-from, not both')
    if args.init_from is None and args.resume_from is None:
        raise ValueError('--init-from is required for fine-tuning a pretrained agent')
    if args.dqn_memory_games and args.algorithm=='ppo':
        raise ValueError('DQN memory games are not applicable to PPO')
    board=TronConfig(args.width,args.height)
    torch.set_num_threads(args.threads)
    source=load(args.init_from,config=board,algorithm=args.algorithm)[0] if args.init_from else (
        _resume_weights(args.resume_from,args.algorithm))
    memory=None
    if args.algorithm=='dqn':
        cfg=StableDQNConfig()
        learner=StableDQNLearner(board,args.seed,cfg,source)
        factory=dqn_factory;architecture=DQN_ARCH
        if args.resume_from is None:
            print(f'Recording {args.dqn_memory_games} games of old-policy experience '
                  '(no gradient updates, no teacher actions)...',flush=True)
            memory=preload_original_dqn_games(learner,board,source,
                 games=args.dqn_memory_games,seed=args.seed)
            print(f'Protected replay: {memory["protected_replay_size"]} transitions',flush=True)
    else:
        cfg=StablePPOConfig()
        learner=StablePPOLearner(board,args.seed,cfg,source)
        factory=ppo_factory;architecture=PPO_ARCH
    cfg_record=dict(**asdict(cfg),architecture=architecture,
                    training_variant='phase56-scripted-opponent-rl',
                    old_policy_memory=memory,
                    initial_checkpoint=str(args.init_from) if args.init_from else None,
                    torch=torch.__version__,threads=args.threads,
                    warm_start='weights_only' if args.init_from else 'full_state_resume')
    summary=run(learner,factory,board,
                RunConfig(seconds=args.seconds,eval_interval=args.eval_interval,
                          eval_pairs=args.eval_pairs,seed=args.seed,stage=args.stage,
                          max_episodes=args.max_episodes),args.out,
                algorithm_config=cfg_record,persist_training_state=True,
                resume_state_path=args.resume_from,selection='strong_balanced')
    if memory is not None:
        (args.out/'protected_memory_summary.json').write_text(json.dumps(memory,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    c=subs.add_parser('train',help='ordinary PPO/DQN reinforcement learning against scripted bots')
    c.add_argument('algorithm',choices=('dqn','ppo'))
    c.add_argument('--stage',choices=('strong_balance','territory_focus'),default='strong_balance')
    c.add_argument('--init-from',type=Path)
    c.add_argument('--resume-from',type=Path)
    c.add_argument('--seconds',type=float,default=1800)
    c.add_argument('--eval-interval',type=float,default=300)
    c.add_argument('--eval-pairs',type=int,default=16)
    c.add_argument('--dqn-memory-games',type=int,default=None,
                   help='default: 160 DQN replay-archive games, 0 to disable')
    c.add_argument('--seed',type=int,default=2026)
    c.add_argument('--threads',type=int,default=1)
    c.add_argument('--width',type=int,default=30)
    c.add_argument('--height',type=int,default=24)
    c.add_argument('--max-episodes',type=int)
    c.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='train':
        if a.dqn_memory_games is None:
            a.dqn_memory_games=160 if a.algorithm=='dqn' else 0
        if a.dqn_memory_games<0 or a.threads<1:
            p.error('memory games must be nonnegative; threads must be positive')
        train(a)


if __name__=='__main__':main()
